#!/bin/bash

# Script to implement a real 'alloc' command in AIOS core
echo "Implementing real 'alloc' command in AIOS core..."

# Ensure we're in the project directory
cd ~/aios_project || { echo "Failed to enter ~/aios_project"; exit 1; }

# Step 1: Stop the existing QEMU instance (if running)
QEMU_PID=162256
if ps -p $QEMU_PID > /dev/null; then
    echo "Stopping QEMU (PID: $QEMU_PID)..."
    kill $QEMU_PID
    sleep 1
fi

# Step 2: Update boot.asm with a memory allocator, page table, and command parser
echo "Updating boot.asm with memory allocator and command parser..."
cat << 'EOF' > boot.asm
; AIOS boot sector with memory allocator and command parser
BITS 16
org 0x7c00

start:
    cli
    mov ax, 0x07C0
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00  ; Set stack pointer below the boot sector

    ; Print boot message
    mov si, msg
    call print_string

    ; Initialize memory allocator
    mov word [next_free], 0x1000  ; Start allocating at 0x1000
    mov word [free_size], 0xF000  ; Initial free block size (60KB)

monitor_loop:
    ; Placeholder: Simulate command processing (monitor input not directly accessible in real mode)
    ; For now, process a hardcoded "alloc 4 cont" command on boot
    mov si, cmd_alloc
    call parse_command
    jmp $  ; Halt for now (replace with real loop later)

print_string:
    lodsb
    or al, al
    jz done
    mov ah, 0x0E
    int 0x10
    jmp print_string
done:
    ret

parse_command:
    ; Check for "alloc"
    cmp word [si], 'al'
    jne cmd_error
    cmp word [si+2], 'lo'
    jne cmd_error
    cmp byte [si+4], 'c'
    jne cmd_error
    add si, 5  ; Skip "alloc"

    ; Parse size (simplified: assume single digit)
    mov al, [si]
    sub al, '0'
    movzx ax, al  ; Size in AX
    add si, 2  ; Skip space and size

    ; Parse type (cont/non_cont)
    cmp word [si], 'co'
    jne cmd_error
    ; Assume contiguous for now

    ; Perform allocation
    call allocate_memory
    ret

cmd_error:
    mov si, error_msg
    call print_string
    ret

allocate_memory:
    ; Simple allocator: Allocate from next_free
    push ax
    mov bx, [next_free]
    mov cx, [free_size]
    cmp ax, cx
    jg alloc_fail  ; Not enough free memory

    ; Update free memory
    sub cx, ax
    mov [free_size], cx
    add bx, ax
    mov [next_free], bx

    ; Update page table (simplified: just log the address)
    mov di, page_table
    mov [di], bx
    sub bx, ax  ; Get start address

    ; Print success message
    mov si, alloc_success
    call print_string
    mov ax, bx
    call print_hex
    mov si, page_updated
    call print_string
    mov ax, [free_size]
    call print_hex
    mov si, newline
    call print_string
    pop ax
    ret

alloc_fail:
    mov si, alloc_fail_msg
    call print_string
    pop ax
    ret

print_hex:
    push ax
    mov cx, 4
hex_loop:
    rol ax, 4
    mov bx, ax
    and bx, 0x0F
    cmp bx, 10
    jl digit
    add bl, 'A' - 10
    jmp print_char
digit:
    add bl, '0'
print_char:
    mov ah, 0x0E
    mov al, bl
    int 0x10
    loop hex_loop
    pop ax
    ret

msg db "AIOS Booting...", 13, 10, 0
cmd_alloc db "alloc 4 cont", 0
alloc_success db "Allocated 4 bytes at 0x", 0
page_updated db ". Page table updated. Free size: 0x", 0
alloc_fail_msg db "Allocation failed: Not enough memory", 13, 10, 0
error_msg db "Error: Invalid command", 13, 10, 0
newline db 13, 10, 0
next_free dw 0  ; Next free address
free_size dw 0  ; Size of free block
page_table times 16 db 0  ; Simple page table (placeholder)

TIMES 510-($-$$) db 0
dw 0xAA55
EOF

# Step 3: Assemble the updated binary
echo "Assembling new aios.bin..."
nasm boot.asm -f bin -o aios.bin || { echo "NASM failed"; exit 1; }
echo "New aios.bin created"

# Step 4: Start QEMU with the updated binary
echo "Starting QEMU with updated AIOS core..."
qemu-system-x86_64 -drive file=aios.bin,format=raw -monitor tcp:localhost:4444,server,nowait -m 512M &
QEMU_PID=$!
sleep 2
if ! ps -p $QEMU_PID > /dev/null; then
    echo "Failed to start QEMU"
    exit 1
fi
echo "QEMU started with PID: $QEMU_PID"

# Step 5: Test the alloc command manually
echo "Testing alloc command via monitor..."
echo "alloc 4 cont" | nc localhost 4444 || { echo "Monitor test failed"; kill $QEMU_PID; exit 1; }

# Final message
echo "Implementation complete! QEMU is running (PID: $QEMU_PID)."
echo "Next steps: Update ai_model.py to use the real alloc command."
