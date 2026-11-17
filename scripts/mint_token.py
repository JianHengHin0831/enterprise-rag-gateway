"""Issue a dev JWT (simulates the upstream SSO). Prints the token only.

python -m scripts.mint_token --tenant acme_corp --role hr --user u_hr_01 [--hours 8]
"""

import argparse
from datetime import timedelta

from app.config import get_settings
from app.schemas import Role
from app.security.auth import issue_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--role", required=True, choices=[r.value for r in Role])
    parser.add_argument("--user", required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    args = parser.parse_args()

    settings = get_settings()
    if args.tenant not in settings.known_tenants:
        parser.error(f"unknown tenant; known: {sorted(settings.known_tenants)}")
    print(
        issue_token(
            tenant_id=args.tenant,
            role=Role(args.role),
            user_id=args.user,
            settings=settings,
            ttl=timedelta(hours=args.hours),
        )
    )


if __name__ == "__main__":
    main()
