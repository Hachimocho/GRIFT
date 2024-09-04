import socket
import sys
import wandb
import os
import pkgutil
import importlib
import subprocess
import json
from trainers import *


if __name__ == "__main__":
    with open("key.txt") as f:
        api_key = f.readline()
        wandb.login(key=api_key)
    with open("serverinfo.json") as f:
        server_info = json.load(f)[0]
        HOST = server_info["host"]
        PORT = int(server_info["port"])
    data = "Requesting sweep."
    print("Config:", wandb.config)
    trainer = globals()[next(iter(wandb.config["trainer"]))](wandb.config["trainer"][next(iter(wandb.config["trainer"]))], wandb.config)
    
    while True:
        # Create a socket (SOCK_STREAM means a TCP socket)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((HOST, PORT))
            sock.sendall(bytes(data + "\n", "utf-8"))

            # Receive data from the server and shut down
            received = str(sock.recv(1024), "utf-8")
            sweep_id, pid, sweep_config = received.split("|")
            # print(sweep_config)
            # sys.exit()
            # with wandb.init(project=pid, config=json.loads(sweep_config)):
            wandb.agent(
                sweep_id=sweep_id, project=pid, count=wandb.config["optimizations_per_sweep"], function=trainer.run
            )
            print("Finished running sweep!")
            sys.exit()