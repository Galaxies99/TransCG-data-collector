import os
import sys
import json
import socket
import struct
from .utils import parse_argv_client


header_buf_size = 4
char_buf_size = 1


def start(dst_addr):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(dst_addr)
    id = struct.unpack('i', s.recv(header_buf_size))[0]
    print("Successfully connect to the server, your client ID is", id)
    return s, id


def exec_cmd(s, cmd):
    # Send commands to the server
    s.send(struct.pack('i', len(cmd)))
    s.sendall(cmd.encode('utf-8'))
    # Receive result
    size = struct.unpack('i', s.recv(header_buf_size))[0]
    res = "".encode('utf-8')
    while len(res) < size:
        res = res + s.recv(char_buf_size)
    size = struct.unpack('i', s.recv(header_buf_size))[0]
    err = "".encode('utf-8')
    while len(err) < size:
        err = err + s.recv(char_buf_size)
    return res.decode('utf-8'), err.decode('utf-8')


def exec_cmd_and_save(s, cmd, res_dir, err_dir=None, display=False):
    res, err = exec_cmd(s, cmd)    
    if display:
        print(res)
    with open(res_dir, "w") as fres:
        fres.write(res)
    if err_dir is not None:
        with open(err_dir, "w") as ferr:
            ferr.write(err)


def close(s):
    s.close()


def exec_client(dst_addr, with_err=False):
    s, _ = start(dst_addr)
    while True:
        # Read commands from input
        cmd = input("NetRelay > ")
        if cmd == "exit":
            break
        # Execute the commands on remote
        res, err = exec_cmd(s, cmd)
        print(res)
        if with_err:
            if len(err) != 0:
                print('[stderr msg]')
                print(err)
        else:
            if len(res) == 0 and len(err) != 0:
                print('[stderr msg]')
                print(err)
    close(s)


def main(argv):
    dst_addr, err = parse_argv_client(argv)
    exec_client(dst_addr, err)


if __name__ == '__main__':
    main(sys.argv[1:])