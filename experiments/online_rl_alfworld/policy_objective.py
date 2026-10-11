"""On-policy REINFORCE with independent same-task leave-one-out baselines."""
def leave_one_out(returns,group_size):
 if group_size<2 or len(returns)%group_size:raise ValueError('Complete groups of at least two required')
 out=[]
 for start in range(0,len(returns),group_size):
  group=returns[start:start+group_size];total=sum(group)
  out.extend(x-(total-x)/(group_size-1) for x in group)
 return out

def trajectory_return(won,steps,log_ratio,beta=0.005,step_cost=0.002):
 return float(won)-step_cost*steps-beta*log_ratio

def policy_loss(log_prob,advantages,num_trajectories):
 # Sum over time, mean over trajectories. Never normalize by trajectory length.
 return -(log_prob*advantages.detach()).sum()/num_trajectories

def matching_rollout_batches(trajectories,advantages,micro_batch):
 # Recreate the exact active-row order and padding groups used during rollout.
 # Trajectory-major regrouping changes padded shapes and can change BF16 logits.
 if len(trajectories)!=len(advantages) or micro_batch<1:raise ValueError('Invalid batch inputs')
 for step in range(max((len(t['transitions']) for t in trajectories),default=0)):
  active=[(t['transitions'][step],advantages[i]) for i,t in enumerate(trajectories) if step<len(t['transitions'])]
  for begin in range(0,len(active),micro_batch):yield active[begin:begin+micro_batch]
