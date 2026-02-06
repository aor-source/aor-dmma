# Contributing to AoR-DMMA

Contributions are welcome. This document describes the requirements for contributing to this project.

## Identity Requirements

**All contributors must use their real identity.** This is non-negotiable.

Before your first contribution, you must:

1. **Set your real name in git config:**
   ```bash
   git config user.name "Your Legal Name"
   git config user.email "your-verified@email.com"
   ```

2. **Your GitHub account must be linked to a verifiable identity.** Pseudonymous accounts, throwaway accounts, and corporate alias accounts (e.g., "a-googler", "user12345") will have their contributions rejected.

3. **Sign your commits.** All commits must be GPG or SSH signed.
   ```bash
   # GPG signing
   git config commit.gpgsign true
   git config user.signingkey YOUR_GPG_KEY_ID

   # Or SSH signing
   git config gpg.format ssh
   git config user.signingkey ~/.ssh/id_ed25519.pub
   git config commit.gpgsign true
   ```

4. **Include a sign-off on every commit** (Developer Certificate of Origin):
   ```bash
   git commit -s -m "Your commit message"
   ```
   This adds a `Signed-off-by: Your Name <email>` line confirming you have the right to submit the code.

## How to Contribute

1. **Fork the repository** to your own GitHub account.

2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes.** Follow existing code style and patterns.

4. **Sign and commit:**
   ```bash
   git commit -s -m "Add description of what you changed and why"
   ```

5. **Open a Pull Request** against `main` with:
   - A clear title describing the change
   - A description explaining the motivation and approach
   - Your **full name** and **GitHub username** in the PR description
   - Reference any related issues

6. **Wait for review.** The repository owner will review all PRs. At least one approving review is required before merge.

## What We Accept

- Expanded AAVE lexicon terms (with citations or sourcing)
- Bug fixes with clear reproduction steps
- New Reinman metrics with mathematical documentation
- Dialect-aware ASR improvements
- Regional variant detection enhancements
- Test coverage improvements
- Documentation improvements

## What We Do NOT Accept

- Commits from unverifiable or pseudonymous accounts
- AI-generated code submitted without human review and sign-off
- Changes that remove or weaken security controls
- Co-authored-by tags for people who did not write or review the code
- Dependency additions without justification
- Changes to branch protection or security policies via PR

## Code of Conduct

- Respect the linguistic and cultural subject matter of this research
- Cite sources when adding to the AAVE lexicon
- Do not submit content that misrepresents or trivializes AAVE
- Engage constructively in code reviews

## Developer Certificate of Origin (DCO)

By signing off on your commits, you certify the following:

```
Developer Certificate of Origin v1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have
    the right to submit it under the open source license indicated in
    the file; or

(b) The contribution is based upon previous work that, to the best of
    my knowledge, is covered under an appropriate open source license
    and I have the right under that license to submit that work with
    modifications, whether created in whole or in part by me, under
    the same open source license; or

(c) The contribution was provided directly to me by some other person
    who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are
    public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Questions

Open an issue or contact the repository owner directly.
