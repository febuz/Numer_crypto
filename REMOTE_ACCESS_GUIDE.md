# Remote Access Guide - Connect from Anywhere

## Prerequisites
1. Router must forward port 51820 (UDP) to 192.168.2.41
2. WireGuard client installed on your remote device

## Connection Steps

### Step 1: Install WireGuard Client

**Windows:**
```bash
# Download from: https://www.wireguard.com/install/
# Or use chocolatey:
choco install wireguard
```

**macOS:**
```bash
# Download from App Store: "WireGuard"
# Or use homebrew:
brew install wireguard-tools
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install wireguard-tools
```

**Mobile:**
- iOS: "WireGuard" from App Store
- Android: "WireGuard" from Google Play

### Step 2: Create Configuration File

Create a file named `server-vpn.conf` with this content:

```ini
[Interface]
PrivateKey = qNhJRZvXr9j7pLVVOtHj3KNY8f9Y3zQBgeTnUUV2F34=
Address = 10.0.0.2/32
DNS = 8.8.8.8

[Peer]
PublicKey = 7yuzO/gv9dQgFl92Td6Miaau90H0VA1m/m6f9Z0BnRk=
Endpoint = 77.166.251.142:51820
AllowedIPs = 192.168.2.0/24, 10.0.0.0/24
PersistentKeepalive = 25
```

### Step 3: Connect to VPN

**GUI Method (Windows/Mac/Mobile):**
1. Import the `server-vpn.conf` file
2. Click "Activate" or toggle the connection ON
3. Wait for "Connected" status

**Command Line (Linux):**
```bash
# Import and start VPN
sudo wg-quick up server-vpn

# Check status
sudo wg show

# You should see something like:
# interface: server-vpn
# peer: 7yuzO/gv9dQgFl92Td6Miaau90H0VA1m/m6f9Z0BnRk=
#   endpoint: 77.166.251.142:51820
#   allowed ips: 192.168.2.0/24, 10.0.0.0/24
```

### Step 4: Test VPN Connection

```bash
# Test if VPN tunnel is working
ping 10.0.0.1

# Should get responses like:
# PING 10.0.0.1 (10.0.0.1) 56(84) bytes of data.
# 64 bytes from 10.0.0.1: icmp_seq=1 ttl=64 time=45.2 ms
```

### Step 5: Access Services

**SSH Access:**
```bash
# Connect via SSH (replace 'username' with your actual username)
ssh username@10.0.0.1

# Example with actual username:
ssh knight2@10.0.0.1
```

**Remote Desktop:**
```bash
# Windows RDP client
mstsc /v:10.0.0.1:3389

# Linux RDP client
rdesktop 10.0.0.1:3389
# or
xfreerdp /v:10.0.0.1:3389 /u:username

# macOS RDP client
# Use "Microsoft Remote Desktop" app, connect to: 10.0.0.1
```

**File Transfer:**
```bash
# Copy files to server
scp localfile.txt username@10.0.0.1:/home/username/

# Copy files from server
scp username@10.0.0.1:/path/to/file.txt ./

# SFTP session
sftp username@10.0.0.1
```

## GPU Access Examples

Once connected via VPN:

**Check GPU Status:**
```bash
ssh username@10.0.0.1 "nvidia-smi"
```

**Run Blender Remotely:**
```bash
# Copy Blender file
scp project.blend username@10.0.0.1:/tmp/

# Render with GPU
ssh username@10.0.0.1 "cd /tmp && blender -b project.blend -o //render_ -f 1 -E CYCLES -- --device CUDA"
```

**Remote Desktop for Blender GUI:**
1. Connect RDP to `10.0.0.1:3389`
2. Open Blender in the remote desktop
3. GPU acceleration automatically available

## Troubleshooting

**VPN Not Connecting:**
```bash
# Check if port 51820 is reachable
nc -u -v 77.166.251.142 51820

# Check router port forwarding is configured:
# External: 51820 UDP → Internal: 192.168.2.41:51820 UDP
```

**Can't Reach 10.0.0.1:**
```bash
# Verify VPN is active
ip addr show | grep 10.0.0.2
# Should show: inet 10.0.0.2/32 scope global wg0

# Check routing
ip route | grep 10.0.0.0
# Should show: 10.0.0.0/24 dev wg0
```

**SSH Connection Refused:**
```bash
# Check if SSH is running on server
ssh -v username@10.0.0.1
# If refused, SSH might not be running or username incorrect
```

## Security Notes

- ✅ **All traffic encrypted** through VPN tunnel
- ✅ **Only VPN port exposed** to internet (51820 UDP)
- ✅ **Services only accessible** via VPN
- ✅ **Strong cryptographic keys** used
- ⚠️  **Keep private keys secure** - never share them
- ⚠️  **Use strong passwords** for SSH/RDP login

## Connection Summary

```
[Your Device] → Internet → [Your Router:51820] → [VPN Server:10.0.0.1] → [Services]
   10.0.0.2         Encrypted Tunnel         10.0.0.1            SSH/RDP/GPU
```

The key point: **10.0.0.1 is only reachable AFTER connecting to VPN first!**