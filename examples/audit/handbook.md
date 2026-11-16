# Lanternfish API operator handbook

Lanternfish is a fictional message-queue API used only to demonstrate `ragwrench audit`.
Every name, limit and price in this handbook is made up.

## Install

Install the command-line client, then point it at your workspace.

```bash
pipx install lanternfish-cli
lanternfish login --workspace demo
lanternfish queues list
```

The installer writes its configuration to `~/.config/lanternfish/config.toml`.

## Rate limits

Limits apply per API key and reset every minute.

| Plan | Requests per minute | Burst | Overage |
| --- | --- | --- | --- |
| Free | 60 | 10 | blocked |
| Team | 600 | 100 | queued |
| Business | 3000 | 500 | billed |
| Enterprise | 12000 | 2000 | negotiated |

## Rotating API keys

1. Create a second key in the dashboard.
2. Deploy the new key next to the old one.
3. Watch the key usage graph until traffic moves.
4. Revoke the old key.
5. Record the rotation date in your runbook.

## Error codes

The API returns one of these codes when a request fails validation or exceeds a limit.

| Code | Meaning | Retry |
| --- | --- | --- |
| LF-400 | Malformed message body | no |
| LF-401 | Missing or revoked key | no |
| LF-429 | Rate limit exceeded | after the reset |
| LF-503 | Queue is rebalancing | with backoff |

## Webhooks

Webhooks deliver each event at least once. Deduplicate on the `event_id` field.

## Support

Open a ticket from the dashboard. Include the request ID from the `X-Lanternfish-Request` header.
