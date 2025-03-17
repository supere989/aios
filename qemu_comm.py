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
