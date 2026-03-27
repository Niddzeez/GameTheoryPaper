# Stage 2 Quick Reference - Trust Dynamics

## What Stage 1 Achieved ✓

✓ **Core Mechanics Validated**
- Threshold cascades work correctly
- Network coordination mechanism functional
- ABM matches analytical predictions (<5% error)

✓ **Key Finding: U-Curve Confirmed**
- Optimal enforcement C* ≈ 0.214
- Initiators: 52% (minimum) → 99% (high C)
- Backfire zone identified: C > 0.26

✓ **Code Infrastructure Ready**
- Modular design (6 clean files)
- Validation framework in place
- Easy to extend with dynamics

---

## What Changes in Stage 2

### Add This Dynamic Loop:

```
Current (Stage 1 - Static):
  T = γT₀/(γ+βC)  [equilibrium, fixed]

New (Stage 2 - Dynamic):
  Each timestep:
    1. Agents cascade based on current T
    2. T updates: T_new = T + dt·[-βC + γ(T₀-T)]
    3. Repeat until (cascade, T) converge
```

### New Phenomena to Observe:

1. **Trust-Cascade Feedback**
   - High resistance → backlash → trust drops
   - Lower trust → lower thresholds → more resistance
   - Positive feedback spiral

2. **Stage 1 Inversion Threshold g***
   - Should emerge as stability boundary
   - g* = γT₀/[β(1+κ)]
   - When C > g*: Trust collapses

3. **Phase Diagrams**
   - (T, x) trajectories
   - Stable vs. unstable regions
   - Oscillations possible

---

## Implementation Checklist

### Step 1: Modify cascade.py
```python
def simulate_cascade_with_trust_dynamics(agents, C, T_initial, ...):
    """
    NEW: Joint cascade-trust simulation
    """
    T = T_initial
    
    while not converged:
        # 1. Cascade step (existing)
        update_agents(agents, C, T, ...)
        
        # 2. Trust update (NEW)
        T_new = T + dt * (-beta*C + gamma*(T_0 - T))
        
        # 3. Check convergence (both cascade AND trust)
        converged = (cascade_stable and abs(T_new - T) < tol)
        
        T = T_new
```

### Step 2: New Validations
- [ ] Does T converge to T* = γT₀/(γ+βC)?
- [ ] Is g* = γT₀/[β(1+κ)] the stability boundary?
- [ ] Do we see trust erosion accelerate cascades?

### Step 3: New Visualizations
- [ ] Trust trajectory: T(t)
- [ ] Phase diagram: (T, x) path
- [ ] Joint convergence: both variables
- [ ] g* boundary visualization

---

## Expected Results

### Scenario A: C < g* (Below Inversion)
- Trust remains stable
- Cascades contained
- System reaches equilibrium

### Scenario B: C > g* (Above Inversion)  
- Trust collapses
- Cascades grow
- Feedback spiral (unstable)

### Scenario C: C ≈ g* (Near Boundary)
- Oscillations possible
- Slow convergence
- Interesting dynamics

---

## Timeline Estimate

| Task | Duration | Deliverable |
|------|----------|-------------|
| Modify cascade.py | 2-3 days | Trust dynamics integrated |
| Test single trajectory | 1 day | T(t) converges to T* |
| Parameter sweep | 2-3 days | g* boundary identified |
| Validation | 2-3 days | Compare to Stage 1 theory |
| Visualization | 1-2 days | Phase diagrams |
| Documentation | 1 day | Stage 2 summary |
| **TOTAL** | **1-2 weeks** | **Stage 2 complete** |

---

## Questions Before Starting Stage 2

1. **Parameter Adjustment?**
   - Current network very dense (100% cascades)
   - Try sparser network to see partial cascades?
   - Or proceed with current setup?

2. **Time Step Size (dt)?**
   - dt = 0.1 (default)
   - Smaller for more accuracy?
   - Test sensitivity?

3. **Convergence Criteria?**
   - |T_new - T| < 0.001 (trust)
   - No cascade changes (behavior)
   - Both required

4. **Initial Conditions?**
   - T(0) = T₀ (start at baseline)
   - T(0) = random perturbation
   - Test multiple initial conditions?

---

## Ready to Proceed?

**Stage 1 Status: ✓ COMPLETE**

All validation criteria met:
- Threshold accuracy: 1.4% error
- U-curve confirmed: C* = 0.214  
- Code modular: Ready to extend
- Documentation: Complete

**Next Step Options:**

A. **Proceed directly to Stage 2**
   - Begin trust dynamics implementation
   - Use current dense network setup
   - ~1-2 weeks to completion

B. **Stage 1 Extensions First**
   - Test sparser networks (see partial cascades)
   - Implement correlated (B, ρ, F)
   - Network comparison (ER vs BA vs WS)
   - +1 week, then Stage 2

C. **Parameter Exploration**
   - Vary α (coordination strength)
   - Vary κ (legitimacy coupling)
   - Sensitivity analysis
   - +3-5 days, then Stage 2

**Recommendation:** Option A (proceed to Stage 2)
- Stage 1 core validated
- Extensions can wait until Stage 4
- Dynamic trust is the next key mechanism

---

## Your Decision

What would you like to do next?

1. ✓ **Start Stage 2 immediately** (add trust dynamics)
2. Explore Stage 1 extensions first (sparser networks, etc.)
3. Parameter sensitivity analysis
4. Other direction?

Let me know and I'll begin implementation!
