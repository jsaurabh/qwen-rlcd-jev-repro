"""Dependency-free synchronous clients. URLs include /v1, not the resource name."""
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request


class APIError(RuntimeError):
    def __init__(self, status, request_id=None):
        self.status, self.request_id = status, request_id
        super().__init__(f'Decision API HTTP {status}' + (f' (request {request_id})' if request_id else ''))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a bearer credential to a redirected host.


class _Client:
    def __init__(self, *, base_url, api_key=None, timeout=90, max_retries=0):
        url = urllib.parse.urlsplit(base_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.query or url.fragment or url.username:
            raise ValueError('base_url must be an HTTP(S) origin and path, without credentials or query')
        if url.scheme == 'http' and url.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('Use HTTPS except for loopback development')
        if not math.isfinite(timeout) or timeout <= 0 or type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError('Require positive timeout and 0–5 retries')
        self.base_url, self.api_key = base_url.rstrip('/'), api_key
        self.timeout, self.max_retries = timeout, max_retries
        self._opener = urllib.request.build_opener(_NoRedirect())

    def _post(self, resource, body):
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        data = json.dumps(body, allow_nan=False).encode()
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(self.base_url + resource, data=data, headers=headers, method='POST')
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    result = json.load(response)
                if not isinstance(result, dict):
                    raise ValueError('Expected a JSON response object')
                return result
            except urllib.error.HTTPError as exc:
                status, request_id = exc.code, exc.headers.get('x-request-id')
                retry_after = exc.headers.get('Retry-After')
                exc.close()
                if status not in (429, 500, 502, 503, 504, 529) or attempt == self.max_retries:
                    raise APIError(status, request_id) from None
                delay = min(2 ** attempt, 8)
                if retry_after:
                    try:
                        seconds = float(retry_after)
                        if math.isfinite(seconds): delay = min(max(seconds, 0), 60)
                    except ValueError:
                        pass
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError):
                # An ambiguous network failure may already have incurred a charge.
                raise RuntimeError('Decision API connection failed; request was not retried') from None


class DecisionsClient(_Client):
    def __init__(self, *, base_url='http://127.0.0.1:8000/v1', **kwargs):
        super().__init__(base_url=base_url, **kwargs)

    def create(self, *, model, input, questions, safety_identifier=None):
        body = {'model': model, 'input': input, 'questions': questions}
        if safety_identifier is not None: body['safety_identifier'] = safety_identifier
        return self._post('/decisions', body)


class SystemOneClient(_Client):
    def __init__(self, *, base_url='https://api.typesafe.ai/v1', **kwargs):
        super().__init__(base_url=base_url, **kwargs)

    def create(self, *, model, state, questions):
        return self._post('/systemone', {'model': model, 'state': state, 'questions': questions})

    def from_decisions(self, *, model, input, questions):
        """Translate the documented text subset, preserving native usage/confidence."""
        from .protocol import decisions_to_systemone, systemone_to_decisions
        body, mapping = decisions_to_systemone({'model': model, 'input': input, 'questions': questions})
        return systemone_to_decisions(self.create(**body), mapping)
