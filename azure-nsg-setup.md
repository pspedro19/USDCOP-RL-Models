# Azure Network Security Group Configuration

## Problem Diagnosed
- VM: `DemoForex` in Resource Group: `DemoForex_group` (East US)
- External IP: `48.216.199.139`
- Port 22 (SSH): ✅ ACCESSIBLE
- Port 80 (HTTP): ❌ BLOCKED/FILTERED
- All Docker containers: ✅ HEALTHY
- Internal connectivity: ✅ WORKING

## Required Azure Commands

### 1. Install Azure CLI (if not available)
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### 2. Login to Azure
```bash
az login
```

### 3. Find Network Security Group
```bash
az network nsg list --resource-group DemoForex_group --output table
```

### 4. Check current NSG rules
```bash
# Replace NSG_NAME with actual NSG name from step 3
az network nsg rule list --resource-group DemoForex_group --nsg-name NSG_NAME --output table
```

### 5. Add HTTP/HTTPS rules
```bash
# Allow HTTP (port 80)
az network nsg rule create \
  --resource-group DemoForex_group \
  --nsg-name NSG_NAME \
  --name Allow-HTTP \
  --protocol Tcp \
  --priority 1000 \
  --destination-port-ranges 80 \
  --source-address-prefixes '*' \
  --access Allow

# Allow HTTPS (port 443)
az network nsg rule create \
  --resource-group DemoForex_group \
  --nsg-name NSG_NAME \
  --name Allow-HTTPS \
  --protocol Tcp \
  --priority 1001 \
  --destination-port-ranges 443 \
  --source-address-prefixes '*' \
  --access Allow
```

### 6. Alternative: Web Portal Method
1. Go to: https://portal.azure.com
2. Navigate to: DemoForex_group > DemoForex VM > Networking
3. Add inbound rules for ports 80 and 443

## Current System Status
- ✅ All 15 Docker containers running and healthy
- ✅ Nginx configuration valid and updated
- ✅ Internal ports listening on 0.0.0.0
- ✅ Server routing configured correctly
- ❌ Azure NSG blocking external HTTP/HTTPS access

**Next Step:** Execute Azure commands above to open ports 80 and 443.