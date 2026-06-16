from __future__ import annotations

from collections.abc import Callable

from fastapi import Header, HTTPException, Request, status
from jose import JWTError, jwt

from scrum_master_assistant.models.auth import AuthContext, AuthenticatedUser, Role
from scrum_master_assistant.models.config import AppSettings


def _parse_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def build_auth_context(settings: AppSettings) -> AuthContext:
    return AuthContext(
        saml_enabled=settings.auth_mode == "saml_proxy",
        oidc_enabled=settings.auth_mode == "oidc_jwt",
        trusted_header_enabled=settings.auth_mode in {"trusted_headers", "saml_proxy"},
        required_roles={
            "dashboard": [Role.viewer.value],
            "stories": [Role.viewer.value],
            "scan": [Role.security_analyst.value, Role.platform_admin.value],
            "publish": [Role.jira_publisher.value, Role.platform_admin.value],
        },
    )


def build_auth_dependency(settings: AppSettings) -> Callable:
    async def get_current_user(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> AuthenticatedUser:
        if not settings.auth_required:
            return AuthenticatedUser(subject="anonymous", display_name="Anonymous", roles={Role.platform_admin}, auth_source="disabled")

        if settings.auth_mode in {"trusted_headers", "saml_proxy"}:
            subject = request.headers.get(settings.auth_trusted_subject_header)
            if not subject:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing trusted identity header")
            groups = _parse_csv(request.headers.get(settings.auth_trusted_groups_header))
            role_names = _parse_csv(request.headers.get(settings.auth_trusted_roles_header))
            mapped_roles = {settings.auth_role_map[group] for group in groups if group in settings.auth_role_map}
            all_roles = {Role(role) for role in role_names | mapped_roles if role in {item.value for item in Role}}
            return AuthenticatedUser(
                subject=subject,
                email=request.headers.get(settings.auth_trusted_email_header),
                display_name=request.headers.get(settings.auth_trusted_name_header),
                groups=groups,
                roles=all_roles or {Role.viewer},
                auth_source=settings.auth_mode,
            )

        if settings.auth_mode == "oidc_jwt":
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
            token = authorization.split(" ", 1)[1]
            if not settings.auth_jwt_secret:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="JWT secret not configured")
            try:
                payload = jwt.decode(
                    token,
                    settings.auth_jwt_secret,
                    algorithms=["HS256"],
                    audience=settings.auth_oidc_audience,
                    issuer=settings.auth_oidc_issuer,
                )
            except JWTError as exc:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
            role_names = set(payload.get("roles", []))
            groups = set(payload.get("groups", []))
            mapped_roles = {settings.auth_role_map[group] for group in groups if group in settings.auth_role_map}
            all_roles = {Role(role) for role in role_names | mapped_roles if role in {item.value for item in Role}}
            return AuthenticatedUser(
                subject=str(payload.get("sub")),
                email=payload.get("email"),
                display_name=payload.get("name"),
                groups=groups,
                roles=all_roles or {Role.viewer},
                auth_source="oidc_jwt",
            )

        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unsupported auth mode")

    return get_current_user
