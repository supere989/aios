; AIOS boot sector with enhanced memory management and command parsing
BITS 16
org 0x7c00

start:
    cli
    mov ax, 0x07C0
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00  ; Set stack pointer below the boot sector

    ; Initialize memory manager
    call init_memory_manager
    
    ; Print boot message
    mov si, msg_boot
    call print_string
    
    ; Enter command loop
    jmp command_loop

init_memory_manager:
    ; Initialize memory tracking table
    mov word [next_free], 0x1000
    mov word [free_size], 0xF000
    mov di, mem_table
    mov cx, MAX_BLOCKS
    xor ax, ax
clear_table:
    mov [di], ax      ; Clear block address
    mov [di+2], ax    ; Clear block size
    add di, 4
    loop clear_table
    ret

command_loop:
    mov si, prompt
    call print_string
    
    ; Get command (simplified for boot sector)
    mov si, buffer
    call parse_command
    jmp command_loop

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
    ; Check command type
    mov di, cmd_table
cmd_check:
    mov si, buffer
    push di
    movzx cx, byte [di]   ; Get command length
    rep cmpsb
    pop di
    je cmd_found
    add di, 8             ; Next command entry
    cmp byte [di], 0
    jne cmd_check
    jmp cmd_error

cmd_found:
    add di, 4         ; Point to handler address
    jmp word [di]     ; Jump to handler

cmd_alloc:
    call parse_number
    call allocate_memory
    ret

cmd_dealloc:
    call parse_number
    call deallocate_memory
    ret

cmd_meminfo:
    call display_memory_info
    ret

cmd_memcopy:
    call parse_two_numbers
    call copy_memory
    ret

cmd_error:
    mov si, error_msg
    call print_string
    ret

allocate_memory:
    ; Input: AX = requested size
    push ax
    mov bx, [next_free]
    mov cx, [free_size]
    
    ; Check if enough memory
    cmp ax, cx
    jg alloc_fail
    
    ; Find free slot in table
    mov di, mem_table
    mov cx, MAX_BLOCKS
find_slot:
    cmp word [di], 0
    je slot_found
    add di, 4
    loop find_slot
    jmp alloc_fail     ; No free slots
    
slot_found:
    pop ax
    mov [di], bx       ; Store block address
    mov [di+2], ax     ; Store block size
    
    ; Update free memory
    sub [free_size], ax
    add [next_free], ax
    
    ; Print success
    mov si, msg_alloc_success
    call print_string
    mov ax, bx
    call print_hex
    ret

alloc_fail:
    mov si, msg_error
    call print_string
    pop ax
    ret

deallocate_memory:
    ; Input: AX = address to deallocate
    mov di, mem_table
    mov cx, MAX_BLOCKS
find_block:
    cmp [di], ax
    je block_found
    add di, 4
    loop find_block
    jmp dealloc_fail
    
block_found:
    mov bx, [di+2]     ; Get block size
    add [free_size], bx
    xor ax, ax
    mov [di], ax       ; Clear table entry
    mov [di+2], ax
    mov si, msg_dealloc_success
    call print_string
    ret

dealloc_fail:
    mov si, msg_error
    call print_string
    ret

copy_memory:
    ; AX = source, BX = dest
    push ds
    push es
    mov cx, 128        ; Copy 128 bytes (adjustable)
    mov si, ax
    mov di, bx
    rep movsb
    pop es
    pop ds
    ret

display_memory_info:
    mov si, msg_meminfo
    call print_string
    
    mov ax, [free_size]
    call print_hex
    
    mov si, msg_allocated
    call print_string
    
    ; Display allocated blocks
    mov di, mem_table
    mov cx, MAX_BLOCKS
display_blocks:
    cmp word [di], 0
    je next_block
    mov ax, [di]
    call print_hex
    mov al, ':'
    mov ah, 0x0E
    int 0x10
    mov ax, [di+2]
    call print_hex
    mov si, msg_bytes
    call print_string
next_block:
    add di, 4
    loop display_blocks
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

; Messages and data
msg_boot db "AIOS Enhanced Boot Loader", 13, 10, 0
prompt db "> ", 0
msg_alloc_success db "Allocated at 0x", 0
msg_dealloc_success db "Memory deallocated", 13, 10, 0
msg_meminfo db "Memory Info - Free: 0x", 0
msg_allocated db " Allocated blocks: ", 13, 10, 0
msg_bytes db " bytes", 13, 10, 0
msg_error db "Error: Invalid command", 13, 10, 0

; Command table
cmd_table:
    db 5, "alloc", cmd_alloc
    db 7, "dealloc", cmd_dealloc
    db 7, "meminfo", cmd_meminfo
    db 7, "memcopy", cmd_memcopy
    db 0  ; End of table

; Data section
buffer times 64 db 0
next_free dw 0
free_size dw 0
MAX_BLOCKS equ 32
mem_table times (MAX_BLOCKS * 4) db 0  ; 32 entries of 4 bytes each

TIMES 510-($-$$) db 0
dw 0xAA55
