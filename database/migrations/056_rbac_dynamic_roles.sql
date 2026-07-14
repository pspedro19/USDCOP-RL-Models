-- ============================================================================
-- Migration 056 — Dynamic RBAC (CTR-RBAC-001 extension)
-- ============================================================================
-- Makes role→permission mapping and per-user permission overrides DATA, so the
-- admin "Roles y vistas" console can edit them at runtime. The rbac.contract.ts
-- route matrices + PERMISSIONS/ROLES enums stay STATIC (the rbac:check grep
-- invariant + deny-by-default floor). Only role→permission and per-user grants
-- become dynamic; the static ROLE_PERMISSIONS remains the seed + fallback.
--
-- Enforcement: effective permissions are baked into the JWT at login (all mint
-- paths) and re-derived by lib/auth/rbac-resolver.ts; middleware stamps them on
-- x-user-perms for the handlers. Empty/absent table ⇒ static fallback (never open).
-- Idempotent: safe to re-run on cold boot.
-- ============================================================================

-- Known permission vocabulary (mirror of rbac.contract.ts PERMISSIONS). A CHECK,
-- not an FK, so it stays in lockstep with the code enum without a roles table.
CREATE TABLE IF NOT EXISTS rbac_role_permissions (
    role        TEXT NOT NULL,
    permission  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (role, permission),
    CONSTRAINT rbac_role_perm_valid CHECK (permission IN (
        'research:read','research:propose','approval:vote','signals:read','forecast:read',
        'analysis:read','execution:self','execution:global','market:read','admin:all'
    ))
);

CREATE TABLE IF NOT EXISTS rbac_user_overrides (
    user_id     UUID NOT NULL,
    permission  TEXT NOT NULL,
    effect      TEXT NOT NULL,
    reason      TEXT,
    updated_by  TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, permission),
    CONSTRAINT rbac_user_override_effect CHECK (effect IN ('grant','deny')),
    CONSTRAINT rbac_user_override_valid CHECK (permission IN (
        'research:read','research:propose','approval:vote','signals:read','forecast:read',
        'analysis:read','execution:self','execution:global','market:read','admin:all'
    ))
);

CREATE INDEX IF NOT EXISTS idx_rbac_user_overrides_user ON rbac_user_overrides (user_id);

-- Seed role→permission from the static ROLE_PERMISSIONS matrix. ON CONFLICT DO
-- NOTHING so an admin's later edits are never clobbered by a re-run.
INSERT INTO rbac_role_permissions (role, permission) VALUES
    -- admin: all ten
    ('admin','research:read'),('admin','research:propose'),('admin','approval:vote'),
    ('admin','signals:read'),('admin','forecast:read'),('admin','analysis:read'),
    ('admin','execution:self'),('admin','execution:global'),('admin','market:read'),
    ('admin','admin:all'),
    -- developer
    ('developer','research:read'),('developer','research:propose'),('developer','forecast:read'),
    ('developer','analysis:read'),('developer','signals:read'),('developer','market:read'),
    -- subscriber
    ('subscriber','signals:read'),('subscriber','forecast:read'),('subscriber','analysis:read'),
    ('subscriber','execution:self'),('subscriber','market:read'),
    -- free
    ('free','forecast:read'),('free','analysis:read')
ON CONFLICT (role, permission) DO NOTHING;
