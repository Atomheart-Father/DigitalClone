## Description
<!-- Briefly describe the changes and their purpose -->

## Type of Change
- [ ] feat: New feature
- [ ] fix: Bug fix
- [ ] docs: Documentation update
- [ ] style: Code style/formatting
- [ ] refactor: Code refactoring
- [ ] test: Adding/Updating tests
- [ ] chore: Maintenance/Configuration

## Checklist (Pre-Merge Self-Review)

### Functional Verification
- [ ] All deliverables from the iteration requirements are implemented
- [ ] Manual acceptance tests pass:
  - [ ] Chat vs Reasoner routing works correctly
  - [ ] Tool calling (calculator, datetime) functions properly
  - [ ] AskUser closed-loop mechanism works without deadlocks
  - [ ] Conversation logs are generated in data/logs/
  - [ ] MockClient works when DEEPSEEK_API_KEY is missing

### Code Quality
- [ ] Tool JSON Schema validation passes
- [ ] Error messages are human-readable
- [ ] AskUser flow is tested and closed-loop (no deadlocks)
- [ ] No sensitive information in logs (proper sanitization)
- [ ] All critical paths have exception handling and logging
- [ ] Code comments and README are updated

### Architecture Compliance
- [ ] Follows message_types protocol consistently
- [ ] Tool return protocol is standardized (value/error fields)
- [ ] LLM interface and tool registry are properly decoupled
- [ ] State machine for AskUser is correctly implemented with limits

### Testing & Validation
- [ ] `pytest -q` passes all tests
- [ ] `python -m backend.cli_app` runs example session successfully
- [ ] New/modified functions have docstrings, type annotations, logging, and error handling
- [ ] Dependencies, startup commands, README snippets, and examples are updated

## Risk Assessment
<!-- Describe any potential risks and mitigation strategies -->

### Risks:
- [ ] Breaking changes to existing functionality
- [ ] Performance degradation
- [ ] Security vulnerabilities
- [ ] Dependency conflicts

### Mitigation:
- [ ] Backward compatibility maintained
- [ ] Performance benchmarks added
- [ ] Security review completed
- [ ] Dependency versions pinned

## Testing Results
<!-- Include test output, coverage reports, or manual test results -->

```
# Test Results
pytest output:
...

Manual test scenarios:
1. Simple chat: "今天天气不错，讲个笑话" → routes to chat
2. Complex task: "给我做个两周学习计划..." → routes to reasoner
3. Tool call: "帮我算 (12+7)*3^2" → calculator tool works
4. AskUser loop: "帮我规划一下周末出游" → asks questions → provides complete plan
```

## Rollback Plan
<!-- How to revert these changes if needed -->

### Rollback Steps:
1. If issues arise, revert the merge commit: `git revert <merge-commit-hash>`
2. Alternative: Create hotfix branch from previous stable commit
3. Monitor logs and metrics for 24 hours post-deployment

### Rollback Validation:
- [ ] Previous functionality restored
- [ ] No data loss occurred
- [ ] User sessions can continue normally
