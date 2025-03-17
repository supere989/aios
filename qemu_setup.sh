#!/bin/bash

# Unified script to set up and test QEMU for AIOS
echo "Starting QEMU setup and test for AIOS..."

# Ensure we're in the project directory
cd ~/aios_project || { echo "Failed to enter ~/aios_project"; exit 1; }

# Step 1: Create the placeholder AIOS binary
echo "Creating placeholder AIOS binary (aios.bin)..."
cat << 'EOF' > boot.asm
; Simple boot sector for QEMU
BITS 16
org 0x7c00

start:
    cli                 ; Disable interrupts
    mov ax, 0x07C0      ; Set up segment registers
    mov ds, ax
    mov es, ax
    mov si, msg         ; Point to message
    call print_string   ; Print a test message
    hlt                 ; Halt
    jmp $               ; Infinite loop

print_string:
    lodsb               ; Load byte from SI into AL
    or al, al           ; Check for null terminator
    jz done             ; If zero, exit
    mov ah, 0x0E        ; BIOS teletype function
    int 0x10            ; Print character
    jmp print_string    ; Next character
done:
    ret

msg db "AIOS Booting...", 0

TIMES 510-($-$$) db 0   ; Pad to 510 bytes
dw 0xAA55               ; Boot signature
EOF

# Assemble the binary
nasm boot.asm -f bin -o aios.bin || { echo "NASM failed"; exit 1; }
echo "Placeholder binary created: aios.bin"

# Step 2: Launch QEMU in the background with the binary as a disk image
echo "Launching QEMU with monitor interface..."
qemu-system-x86_64 -drive file=aios.bin,format=raw -monitor tcp:localhost:4444,server,nowait -m 512M -nographic &
QEMU_PID=$!
sleep 2  # Give QEMU a moment to start

# Check if QEMU is running
if ! ps -p $QEMU_PID > /dev/null; then
    echo "QEMU failed to start"
    exit 1
fi
echo "QEMU running with PID: $QEMU_PID"

# Step 3: Create the communication layer script
echo "Creating qemu_comm.py..."
cat << 'EOF' > qemu_comm.py
import socket
import time

class QEMUComm:
    def __init__(self, host="localhost", port=4444):
        self.host = host
        self.port = port

    def send_command(self, cmd):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        sock.send((cmd + "\n").encode())
        time.sleep(0.1)  # Wait for response
        response = sock.recv(4096).decode().strip()
        sock.close()
        return response

if __name__ == "__main__":
    comm = QEMUComm()
    print("Sending 'info registers':")
    print(comm.send_command("info registers"))
    print("Sending 'info memory':")
    print(comm.send_command("info memory"))
EOF

# Step 4: Test the communication
echo "Testing communication with QEMU..."
python3 qemu_comm.py || { echo "Communication test failed"; kill $QEMU_PID; exit 1; }

# Cleanup: Keep QEMU running for now, but show how to stop it
echo "Test complete! QEMU is still running (PID: $QEMU_PID)."
echo "To stop QEMU, run: kill $QEMU_PID"
echo "Next steps: Build AI model I/O and integrate LM-Studio training."
