# =============================================================================
# HashiCorp Vault Configuration
# =============================================================================
# This config is for production use. Development mode uses in-memory storage.
# To switch to production, update docker-compose to use:
#   command: server -config=/vault/config/vault-config.hcl
# =============================================================================

# Storage backend - File storage for single-node deployment
storage "file" {
  path = "/vault/file"
}

# Listener configuration
listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 1  # Enable TLS in production with proper certificates
}

# API address for clients
api_addr = "http://127.0.0.1:8200"

# Disable memory locking (required for Docker without IPC_LOCK capability)
disable_mlock = false

# UI configuration
ui = true

# Log level
log_level = "info"
