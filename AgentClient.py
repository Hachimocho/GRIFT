import socket
import sys
import wandb
import os
import pkgutil
import importlib
import subprocess
from utils.import_utils import load_class_from_globals
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
        with SSHTunnelForwarder(
            ('nsf-gpu.main.ad.rit.edu', 22),  # Replace with your SSH server details
            ssh_username='brg2890',
            ssh_password=PASSKEY,
            remote_bind_address=('localhost', 9998)  # Replace with the desired remote address and port
        ) as server:
            # Connect to the remote server via the SSH tunnel
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(('localhost', server.local_bind_port))
                sock.sendall(bytes(data + "\n", "utf-8"))
                data = sock.recv(1024)
                print('Received:', data.decode())
                # Receive sweep config from the server
                received = str(sock.recv(1024), "utf-8")
                sweep_id, pid, sweep_config = received.split("|")
                # Print out config for user review
                print("Config:", wandb.config)
                print("sweep config", sweep_config)
                
                try:   
                    #data, dataloader, dataset, graph, manager, model, trainer, traversal = sweep_config["parameters"]["name"].split("_")
                    #test_dataset, test_dataloader, test_traversal = globals()[next(iter(sweep_config["parameters"]["test_config"]["value"]["dataset"].keys()))](), sweep_config["parameters"]["test_config"]["value"]["dataloader"].keys()[0], sweep_config["parameters"]["test_config"]["value"]["traversal"].keys()[0]
                    trainer = globals()[next(iter(wandb.config["trainer"]))](wandb.config["trainer"][next(iter(wandb.config["trainer"]))], wandb.config)
                    wandb.agent(
                        sweep_id=sweep_id, project=pid, count=wandb.config["optimizations_per_sweep"], function=trainer.run
                    )
                    print("Finished running sweep!")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        sock.sendall(b"GPU out of memory\n")
                        print("GPU out of memory, reporting to server")
                    else:
                        raise e
                
                