## GitHub Actions → Orchestrator CI Reporting

Add this workflow to your repo so CI results are POSTed to the orchestrator.

```yaml
name: CI
on:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: make ci

  report:
    needs: [test]
    if: ${{ always() }}
    runs-on: ubuntu-latest
    steps:
      - name: Report to orchestrator
        env:
          ORCH_URL: ${{ secrets.ORCH_URL }}
          ORCH_TOKEN: ${{ secrets.ORCH_TOKEN }}
          NTFY_TOPIC: ${{ secrets.NTFY_TOPIC }}
        run: |
          curl -sS -X POST "$ORCH_URL/webhooks/ci" \
            -H "Authorization: Bearer $ORCH_TOKEN" \
            -H "Content-Type: application/json" \
            -d @- <<'JSON'
          {
            "repo": "${{ github.repository }}",
            "pr": ${{ github.event.pull_request.number }},
            "conclusion": "${{ job.status }}",
            "run_id": "${{ github.run_id }}",
            "summary_url": "https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }}",
            "ntfy_topic": "${NTFY_TOPIC}"
          }
          JSON
```

See also: Cursor Background Agents API – Overview: https://docs.cursor.com/en/background-agent/api/overview


