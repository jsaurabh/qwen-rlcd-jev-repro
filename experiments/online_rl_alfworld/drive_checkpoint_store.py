"""Mount-free Drive checkpoint storage. Credentials stay outside the repository.

Requires google-api-python-client and google-auth-httplib2. Files are uploaded
resumably and checksum-verified before a remote completion manifest is written.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import datetime
from pathlib import Path, PurePosixPath

import httplib2
from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload, MediaIoBaseDownload

SCOPE = "https://www.googleapis.com/auth/drive.file"
MARKER = "DRIVE_COMPLETE.json"


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(chunk); sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


class DriveStore:
    def __init__(self, credentials_file, folder_id):
        info = json.loads(Path(credentials_file).read_text())
        if info.get("type") == "access_token_only":
            credentials = Credentials(token=info["token"],
                expiry=datetime.datetime.fromisoformat(info["expiry"].removesuffix("Z")),
                quota_project_id=info.get("quota_project_id"))
        else:
            credentials = Credentials.from_authorized_user_info(info, scopes=[SCOPE])
        transport = httplib2.Http(timeout=120)
        # Drive uses 308 for incomplete resumable uploads, not HTTP redirects.
        transport.redirect_codes = transport.redirect_codes - {308}
        self.api = build("drive", "v3", http=AuthorizedHttp(credentials, http=transport),
                         cache_discovery=False)
        self.folder_id = folder_id

    def children(self, parent):
        result, token = [], None
        while True:
            page = self.api.files().list(q=f"'{parent}' in parents and trashed = false",
                fields="nextPageToken,files(id,name,mimeType,size,md5Checksum)",
                pageSize=1000, pageToken=token).execute(num_retries=5)
            result.extend(page.get("files", [])); token = page.get("nextPageToken")
            if not token:
                return result

    def folder(self, name, parent=None):
        parent = parent or self.folder_id
        matches = [x for x in self.children(parent) if x["name"] == name]
        if len(matches) > 1:
            raise RuntimeError(f"Ambiguous Drive folder: {name}")
        if matches:
            if matches[0]["mimeType"] != "application/vnd.google-apps.folder":
                raise RuntimeError(f"Drive path is not a folder: {name}")
            return matches[0]["id"]
        return self.api.files().create(body={"name": name, "parents": [parent],
            "mimeType": "application/vnd.google-apps.folder"}, fields="id").execute(num_retries=5)["id"]

    def upload_file(self, path, parent):
        path = Path(path)
        md5, sha = hashes(path)
        existing = [x for x in self.children(parent) if x["name"] == path.name]
        if existing:
            if len(existing) != 1 or existing[0].get("md5Checksum") != md5:
                raise RuntimeError(f"Refusing to overwrite different Drive data: {path.name}")
            result = existing[0]
        else:
            request = self.api.files().create(body={"name": path.name, "parents": [parent]},
                media_body=MediaFileUpload(str(path), chunksize=8*1024*1024, resumable=True),
                fields="id,size,md5Checksum")
            result = None
            while result is None:
                _, result = request.next_chunk(num_retries=5)
        if result.get("md5Checksum") != md5 or int(result["size"]) != path.stat().st_size:
            raise RuntimeError(f"Drive checksum/size mismatch: {path.name}")
        return {"id": result["id"], "size": path.stat().st_size, "md5": md5, "sha256": sha}

    def read_json(self, file_id):
        return json.loads(self.api.files().get_media(fileId=file_id).execute(num_retries=5))

    def upload_checkpoint(self, path):
        path = Path(path)
        if not (path / "COMPLETE.json").is_file():
            raise RuntimeError("Checkpoint is not locally complete")
        files = sorted(p for p in path.rglob("*") if p.is_file())
        if any(p.is_symlink() for p in path.rglob("*")):
            raise ValueError("Checkpoint symlinks are not allowed")
        if any(p.name == MARKER for p in files):
            raise ValueError("Reserved marker in source checkpoint")
        remote = self.folder(path.name)
        previous = [x for x in self.children(remote) if x["name"] == MARKER]
        if previous:
            if len(previous) != 1: raise RuntimeError("Duplicate completion markers")
            manifest = self.read_json(previous[0]["id"])
            current = {p.relative_to(path).as_posix(): hashes(p)[1] for p in files}
            if current != {k:v["sha256"] for k,v in manifest["files"].items()}:
                raise RuntimeError("Completed remote checkpoint differs from local checkpoint")
            return {"folder_id":remote,"files":len(files),"already_verified":True}
        folders = {".": remote}
        entries = {}
        for p in files:
            rel = p.relative_to(path)
            parent = remote
            for n in range(1, len(rel.parts)):
                key = PurePosixPath(*rel.parts[:n]).as_posix()
                if key not in folders: folders[key] = self.folder(rel.parts[n-1], parent)
                parent = folders[key]
            entries[rel.as_posix()] = self.upload_file(p, parent)
        payload = json.dumps({"version":1,"checkpoint":path.name,"files":entries},sort_keys=True).encode()
        created = self.api.files().create(body={"name":MARKER,"parents":[remote]},
            media_body=MediaIoBaseUpload(io.BytesIO(payload),mimetype="application/json"),
            fields="id,md5Checksum").execute(num_retries=5)
        if created.get("md5Checksum") != hashlib.md5(payload).hexdigest():
            raise RuntimeError("Completion marker checksum mismatch")
        return {"folder_id":remote,"files":len(files),"already_verified":False}

    def restore(self, checkpoint_folder_id, destination):
        destination = Path(destination)
        if destination.exists(): raise FileExistsError(destination)
        markers = [x for x in self.children(checkpoint_folder_id) if x["name"] == MARKER]
        if len(markers) != 1: raise RuntimeError("No unique verified completion manifest")
        manifest = self.read_json(markers[0]["id"])
        temporary = destination.with_name(destination.name + ".download-partial")
        temporary.mkdir(parents=True,exist_ok=False)
        for name, meta in manifest["files"].items():
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts or "\\" in name:
                raise ValueError("Unsafe checkpoint filename")
            path = temporary.joinpath(*rel.parts); path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("wb") as output:
                downloader = MediaIoBaseDownload(output,self.api.files().get_media(fileId=meta["id"]),chunksize=8*1024*1024)
                done = False
                while not done: _,done = downloader.next_chunk(num_retries=5)
            if hashes(path)[1] != meta["sha256"] or path.stat().st_size != meta["size"]:
                raise RuntimeError(f"Restored checkpoint checksum mismatch: {name}")
        if not (temporary/"COMPLETE.json").is_file(): raise RuntimeError("Missing local completion marker")
        temporary.rename(destination)
        return {"restored":str(destination),"files":len(manifest["files"])}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--credentials",default=os.environ.get("DRIVE_CHECKPOINT_CREDENTIALS"))
    parser.add_argument("--folder",required=True)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("mkdir").add_argument("name")
    sub.add_parser("upload").add_argument("path",type=Path)
    restore=sub.add_parser("restore");restore.add_argument("checkpoint_id");restore.add_argument("destination",type=Path)
    args=parser.parse_args()
    if not args.credentials: parser.error("Provide --credentials or DRIVE_CHECKPOINT_CREDENTIALS")
    store=DriveStore(args.credentials,args.folder)
    if args.command == "mkdir": result={"folder_id":store.folder(args.name)}
    elif args.command == "upload": result=store.upload_checkpoint(args.path)
    else: result=store.restore(args.checkpoint_id,args.destination)
    print(json.dumps(result),flush=True)


if __name__ == "__main__": main()
