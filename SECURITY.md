# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| v7.1.x (MLX) | Yes |
| < v7.0 | No |

## Reporting Vulnerabilities

If you discover a security vulnerability in AoR-DMMA, please report it responsibly:

1. **Do NOT open a public issue.**
2. Email: Contact the repository owner directly through GitHub.
3. Include: Description of the vulnerability, steps to reproduce, and potential impact.
4. You will receive a response within 72 hours.

## Repository Security Controls

### Branch Protection (main)

The `main` branch is protected with the following rules:

- **Pull request reviews required** — All changes must go through a PR with at least 1 approving review
- **Stale review dismissal** — Approvals are dismissed when new commits are pushed
- **Code owner review required** — The repository owner must approve all PRs
- **Linear history required** — No merge commits; rebase or squash only
- **No force pushes** — Force push to `main` is permanently disabled
- **No branch deletion** — The `main` branch cannot be deleted
- **Conversation resolution required** — All PR review threads must be resolved before merge

### Contributor Identity Verification

All contributors must:

1. Use their **real, legal name** in git config (not a pseudonym, handle, or corporate alias)
2. Provide their **GitHub username** linked to a verifiable identity
3. Include a DCO sign-off on every commit (see CONTRIBUTING.md)
4. Agree to the Developer Certificate of Origin
5. **Not use VPN, proxy, Tor, or any IP-masking service** when interacting with this repository

Anonymous contributions, bot-authored commits without human sign-off, and pseudonymous accounts (e.g., "a-googler", "dependabot-contributor") are **not accepted**.

Contributors who cannot meet these requirements may submit changes via email instead. See PRIVACY.md for details.

### Network Access Policy

**VPN, proxy, and Tor connections are prohibited** for all repository interactions (clone, push, PR, review, comment). GitHub logs IP addresses on all git operations, and this project relies on those logs for security forensics. Masked IP addresses undermine our ability to investigate unauthorized access.

See PRIVACY.md for the full privacy policy, data handling practices, and the email-based alternative contribution method.

### Audit Trail

- All changes to `main` are tracked through pull requests with full review history
- The GitHub audit log records all collaborator additions, permission changes, and branch protection modifications
- Push events are logged via GitHub Actions (see `.github/workflows/audit-log.yml`)

### What We Do NOT Accept

- Commits from accounts with no verifiable identity
- Co-authored-by tags for individuals who did not review or write the code
- Automated commits from third-party bots without explicit owner authorization
- Force pushes or history rewrites to protected branches
- Contributions from VPN, proxy, Tor, or otherwise masked IP addresses
- Contributions from accounts that cannot be traced to a real person

## Data Handling

AoR-DMMA processes audio files locally. No audio data is transmitted externally. Whisper transcription runs on-device via MLX (Apple Silicon) or locally via PyTorch.

The `aave_lexicon.json` contains linguistic research data only — no personally identifiable information.

## Dependencies

We review dependencies for known vulnerabilities. If you identify a vulnerable dependency, please report it following the process above.
