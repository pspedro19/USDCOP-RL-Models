# Post-Mortem: {INCIDENT_TITLE}

**Incident ID**: INC-{YYYYMMDD}-{NNN}
**Date**: {DATE}
**Duration**: {START_TIME} - {END_TIME} ({DURATION})
**Severity**: P{0-3}
**Status**: {Draft | Reviewed | Closed}

---

## Executive Summary

{One paragraph summary of what happened, the impact, and how it was resolved. This should be understandable by non-technical stakeholders.}

---

## Impact

### Quantitative Impact

| Metric | Value |
|--------|-------|
| Trading Downtime | {X} minutes |
| Missed Trades | {N} trades |
| Erroneous Trades | {N} trades |
| Financial Impact | ${amount} / {description} |
| Users/Systems Affected | {description} |
| Data Loss | {None / description} |

### Qualitative Impact

- {Impact on team confidence/morale}
- {Impact on stakeholder trust}
- {Compliance implications, if any}

---

## Timeline (All Times in COT - Colombia Time)

| Time | Event | Actor |
|------|-------|-------|
| {HH:MM} | First symptom occurred | System |
| {HH:MM} | Alert triggered | Monitoring |
| {HH:MM} | On-call acknowledged | {Name} |
| {HH:MM} | Initial investigation started | {Name} |
| {HH:MM} | {Key discovery or action} | {Name} |
| {HH:MM} | Root cause identified | {Name} |
| {HH:MM} | Mitigation applied | {Name} |
| {HH:MM} | Service restored | {Name} |
| {HH:MM} | Incident closed | {Name} |

---

## Root Cause Analysis

### What Happened?

{Detailed technical explanation of what went wrong. Be specific about the sequence of events and the technical components involved.}

```
{Include relevant code snippets, configurations, or commands if helpful}
```

### Why Did It Happen?

{Analysis of the underlying causes. Use the "5 Whys" technique if appropriate.}

**5 Whys Analysis:**

1. **Why did {symptom} occur?**
   - Because {cause 1}

2. **Why did {cause 1} happen?**
   - Because {cause 2}

3. **Why did {cause 2} happen?**
   - Because {cause 3}

4. **Why did {cause 3} happen?**
   - Because {cause 4}

5. **Why did {cause 4} happen?**
   - Because {root cause}

### Why Wasn't It Caught Earlier?

{Analysis of gaps in monitoring, testing, or process that allowed this to happen.}

- {Gap 1: e.g., "No alert configured for this specific error condition"}
- {Gap 2: e.g., "This code path was not covered by integration tests"}
- {Gap 3: e.g., "Runbook did not include this failure mode"}

---

## Contributing Factors

List all factors that contributed to the incident, even if they weren't the primary cause:

1. **{Factor 1 Title}**
   - Description: {What was the factor}
   - Contribution: {How it contributed to the incident}

2. **{Factor 2 Title}**
   - Description: {What was the factor}
   - Contribution: {How it contributed to the incident}

3. **{Factor 3 Title}**
   - Description: {What was the factor}
   - Contribution: {How it contributed to the incident}

---

## Resolution

### Immediate Fix

{What was done to restore service?}

```bash
# Commands or steps taken
{command 1}
{command 2}
```

### Verification

{How was the fix verified?}

- {Verification step 1}
- {Verification step 2}
- {Verification step 3}

---

## Action Items

| ID | Action | Owner | Priority | Due Date | Status |
|----|--------|-------|----------|----------|--------|
| 1 | {Prevent recurrence action} | {Name} | P{0-2} | {YYYY-MM-DD} | {Open/In Progress/Done} |
| 2 | {Improve detection action} | {Name} | P{0-2} | {YYYY-MM-DD} | {Open/In Progress/Done} |
| 3 | {Documentation update} | {Name} | P{0-2} | {YYYY-MM-DD} | {Open/In Progress/Done} |
| 4 | {Process improvement} | {Name} | P{0-2} | {YYYY-MM-DD} | {Open/In Progress/Done} |
| 5 | {Training/knowledge share} | {Name} | P{0-2} | {YYYY-MM-DD} | {Open/In Progress/Done} |

### Action Item Categories

- **Prevent Recurrence**: Changes to prevent the same issue from happening again
- **Improve Detection**: Better monitoring, alerting, or observability
- **Reduce Impact**: Changes to limit blast radius if similar issues occur
- **Process Changes**: Updates to runbooks, procedures, or policies
- **Training**: Knowledge sharing or skill development needs

---

## Lessons Learned

### What Went Well

- {Positive aspect 1: e.g., "Alert fired within 2 minutes of the issue starting"}
- {Positive aspect 2: e.g., "Rollback procedure worked as documented"}
- {Positive aspect 3: e.g., "Team communication was clear and effective"}

### What Could Be Improved

- {Improvement area 1: e.g., "Alert message could be more actionable"}
- {Improvement area 2: e.g., "Runbook was missing steps for this scenario"}
- {Improvement area 3: e.g., "Too long to escalate to secondary on-call"}

### What We Will Do Differently

- {Specific change 1}
- {Specific change 2}
- {Specific change 3}

---

## Appendix

### A. Relevant Logs

```
{Key log excerpts that illustrate the issue - sanitize any sensitive data}
```

### B. Metrics and Graphs

{Links to relevant Grafana dashboards or embedded screenshots}

- Dashboard: {link}
- Error rate graph: {link or screenshot}
- Latency graph: {link or screenshot}

### C. Configuration Changes

{Any configuration that was changed as part of resolution}

```yaml
# Before
{old_configuration}

# After
{new_configuration}
```

### D. Related Incidents

| Incident ID | Date | Similarity |
|-------------|------|------------|
| INC-{YYYYMMDD}-{NNN} | {Date} | {How related} |

### E. External References

- {Link to relevant documentation}
- {Link to vendor status page at time of incident}
- {Link to related GitHub issue or PR}

---

## Sign-off

### Post-Mortem Author

| Field | Value |
|-------|-------|
| Name | {Author Name} |
| Role | {Author Role} |
| Date Written | {YYYY-MM-DD} |

### Reviewers

| Name | Role | Date Reviewed | Approved |
|------|------|---------------|----------|
| {Name} | {Role} | {YYYY-MM-DD} | Yes/No |
| {Name} | {Role} | {YYYY-MM-DD} | Yes/No |

### Follow-up Schedule

| Date | Review Type | Owner |
|------|-------------|-------|
| {Date} | Action items check | {Name} |
| {Date} | 30-day retrospective | {Name} |
| {Date} | Final close-out | {Name} |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | {Date} | {Name} | Initial draft |
| 0.2 | {Date} | {Name} | Added root cause analysis |
| 1.0 | {Date} | {Name} | Final version after review |

---

*This post-mortem follows the blameless incident analysis methodology. The goal is to learn and improve, not to assign blame. All times are in COT (Colombia Time, UTC-5).*
