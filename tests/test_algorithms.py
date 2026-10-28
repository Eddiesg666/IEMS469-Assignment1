"""Deterministic numerical checks, not a substitute for experimental validation."""
import numpy as np
import torch
from shuttle import (make_mdp, finite_horizon, q_values, value_iteration,
                     policy_iteration)
from pg import discounted_returns, preprocess, CartPolicy, PongPolicy


def test_mdp_probabilities_and_bounds():
    p,c = make_mdp()
    assert p.shape == (201,2,201)
    assert c.shape == (201,2)
    assert np.allclose(p.sum(axis=-1),1)
    assert np.all(p>=0)
    assert c[0,0] == 0
    assert c[20,0] == 40
    assert c[20,1] == 110
    # At 200, the station cap must apply after arrivals.
    assert np.isclose(p[200,0,200],1)
    # At 0, waiting first then all arrival states 1...5 equally likely.
    assert np.allclose(p[0,0,1:6],.2)


def test_mdp_small_deterministic_hand_computation():
    p,c = make_mdp(capacity=2,dispatch_cost=3,holding_cost=1,
                   max_people=3,arrivals=(0,))
    v,pi = finite_horizon(p,c,gamma=.5,horizon=1)
    # Terminal costs = 0. So choose min immediate cost.
    assert np.allclose(v,[0,1,2,3])
    assert np.array_equal(pi,[0,0,0,0]) # Tie at s=3 -> prefer wait
    # For cost-free dispatch all positive queues strictly prefer dispatch.
    p2,c2=make_mdp(capacity=2,dispatch_cost=0,holding_cost=1,
                   max_people=4,arrivals=(0,))
    v2,pol,_,_=value_iteration(p2,c2,gamma=.5)
    assert np.array_equal(pol[1:],[1,1,1,1])
    assert abs(v2[0])<1e-9


def test_bellman_residual_and_agreement():
    p,c=make_mdp(max_people=20)
    v, pol, it, _=value_iteration(p,c,tol=1e-9)
    v2, pol2, it2=policy_iteration(p,c)
    assert np.max(np.abs(v-np.min(q_values(p,c,v,.95),axis=1)))<1e-7
    assert np.allclose(v,v2,atol=1e-7)
    assert np.array_equal(pol,pol2)


def test_discounted_returns_hand_calculation():
    assert np.allclose(discounted_returns([1,2,3],.5),[2.75,3.5,3])
    assert np.allclose(discounted_returns([-1,0,1],.99),[-.0199,.99,1])
    assert discounted_returns([], .95).size == 0
    assert np.allclose(discounted_returns([1,2,3],0),[1,2,3])


def test_preprocess_erase_background_and_binarize():
    x=np.zeros((210,160,3),dtype=np.uint8)
    x[35,0,0]=144
    x[37,2,0]=109
    x[39,4,0]=200
    result=preprocess(x)
    assert result.shape==(80,80)
    assert result.dtype==np.float32
    assert result[0,0]==0 and result[1,1]==0 and result[2,2]==1
    assert set(np.unique(result)) <= {0,1}


def test_policies_have_exactly_two_actions_and_gradients():
    for model,x in [(CartPolicy(),torch.randn(3,4)),
                    (PongPolicy(),torch.randn(3,1,80,80))]:
        logits=model(x)
        assert logits.shape==(3,2)
        loss=-torch.log_softmax(logits,dim=1)[:,0].mean()
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                   and torch.abs(p.grad).sum() > 0 for p in model.parameters())


def test_zero_advantage_has_zero_policy_gradient():
    policy=CartPolicy()
    logits=policy(torch.randn(5,4))
    chosen=torch.log_softmax(logits,dim=1)[:,0]
    loss=-(chosen*torch.zeros(5)).mean()
    loss.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0
               for p in policy.parameters())


def test_policy_gradient_direction_on_controlled_one_step_problem():
    """Positive reward must increase probability of the chosen action."""
    torch.manual_seed(7)
    model=CartPolicy()
    obs=torch.tensor([[0.1,0.0,-0.2,0.0]])
    with torch.no_grad():
        old=torch.softmax(model(obs),dim=-1)[0,1].item()
    opt=torch.optim.SGD(model.parameters(), lr=.005)
    logp=torch.log_softmax(model(obs),dim=-1)[0,1]
    loss=-logp*1.0  # return-to-go=+1, action=1
    opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        new=torch.softmax(model(obs),dim=-1)[0,1].item()
    assert new > old, (old,new)
