# GitHub To VPS Deployment Setup

This project uses GitHub Actions to connect to the VPS over SSH and sync the repository.

## Required GitHub Secrets

- `VPS_HOST`
- `VPS_PORT`
- `VPS_USER`
- `VPS_SSH_KEY`
- `DEPLOY_PATH`

## Recommended Values

- `VPS_HOST`: `dcc835eb3940.vps.myjino.ru`
- `VPS_PORT`: `49449`
- `VPS_USER`: `root`
- `DEPLOY_PATH`: for example `/opt/menu-intelligence`

## Flow

1. GitHub Actions authenticates to the VPS with an SSH private key stored in `VPS_SSH_KEY`.
2. The VPS stores the matching public key in `~/.ssh/authorized_keys`.
3. On push to `main` or manual dispatch, GitHub runs the workflow.
4. The workflow connects to the VPS and executes `ops/remote_sync.sh`.
5. The VPS clones or fast-forwards the public repository.

## Security Notes

- No password is stored in the repository.
- No SSH private key is committed.
- The repository can remain public because the VPS pulls it over HTTPS.
