import socket
import sys
import wandb
import os
import pkgutil
import importlib
import subprocess
from trainers import *
from sshtunnel import SSHTunnelForwarder

if __name__ == "__main__":
    # with open("key.txt") as f:
    #     api_key = f.readline()
    #     wandb.init()
    with open("serverinfo.txt") as f:
        HOST = socket.gethostbyaddr(f.readline().strip())[-1][0]
        PORT = int(f.readline())
    data = "Requesting sweep."
    print(HOST, ":", PORT)
    while True:
        # Create a socket (SOCK_STREAM means a TCP socket)
        # with SSHTunnelForwarder(
        #     ('ssh_host', 22),  # Replace with your SSH server details
        #     ssh_username='brg2890',
        #     ssh_password='your_password',
        #     remote_bind_address=('localhost', 8080)  # Replace with the desired remote address and port
        # ) as server:
        #     # Connect to the remote server via the SSH tunnel
        #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        #         sock.connect(('localhost', server.local_bind_port))
        #         sock.sendall(b'Hello, world!')
        #         data = sock.recv(1024)
        #         print('Received:', data.decode())
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Connect to server and send data
            sock.connect((HOST, PORT))
            sock.sendall(bytes(data + "\n", "utf-8"))

            # Receive data from the server and shut down
            received = str(sock.recv(1024), "utf-8")
            sweep_id, pid, sweep_config = received.split("|")
            # print(sweep_config)
            # sys.exit()
            # with wandb.init(project=pid, config=json.loads(sweep_config)):
            print("Config:", wandb.config)
            print("sweep config", sweep_config)
            trainer = globals()[next(iter(wandb.config["trainer"]))](wandb.config["trainer"][next(iter(wandb.config["trainer"]))], wandb.config)
            wandb.agent(
                sweep_id=sweep_id, project=pid, count=wandb.config["optimizations_per_sweep"], function=trainer.run
            )
            print("Finished running sweep!")
            sys.exit()