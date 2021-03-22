import os
import sys
import json
import socket
import struct
import getopt
import threading
import subprocess
from .utils import parse_cmd, parse_argv_relay


header_buf_size = 4
char_buf_size = 1


def exec_conn(conn, addr, id):
    conn.send(struct.pack('i', id))
    print('[Log (ID: %d)] Login from' % id, addr)
    while True:
        # Receive command.
        try:
            size = struct.unpack('i', conn.recv(header_buf_size))[0]
        except Exception:
            print('[Log (ID: %d)] Logout' % id)
            break
        cmd = "".encode('utf-8')
        while(len(cmd) < size):
            cmd = cmd + conn.recv(char_buf_size)
        cmd = cmd.decode('utf-8')
        print('[Log (ID: %d)] (1/3) Receive command' % id, cmd)
        # Execute command and get result.
        with open("netrelay_logs/res" + str(id) + ".dat", "wb") as fout, open("netrelay_logs/errmsg" + str(id) + ".dat", "wb") as ferr:
            try:
                _ = subprocess.call(cmd, stdout=fout, stderr=ferr, shell=True)
            except Exception:
                print('[Log (ID: %d)] Unsupported command' % id)
                ferr.write("Unsupported command.\n".encode('utf-8'))
                fout.write("".encode('utf-8'))
        with open("netrelay_logs/res" + str(id) + ".dat", "rb") as fout:
            res = fout.read()
        with open("netrelay_logs/errmsg" + str(id) + ".dat", "rb") as ferr:
            err = ferr.read()
        print('[Log (ID: %d)] (2/3) Finish executing' % id)
        # Send back result.
        conn.send(struct.pack('i', len(res)))
        conn.sendall(res)
        conn.send(struct.pack('i', len(err)))
        conn.sendall(err)
        print('[Log (ID: %d)] (3/3) Finish data sending' % id)


def exec_relay(src_addr):
    if not os.path.exists('netrelay_logs'):
        os.makedirs('netrelay_logs')
    id_cnt = 0
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(src_addr)
    s.listen(5)

    threads = []
    while True:
        conn, addr = s.accept()
        id_cnt = id_cnt + 1
        id = id_cnt
        t = threading.Thread(target=exec_conn, args=(conn, addr, id))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    s.close()


def main(argv):
    exec_relay(parse_argv_relay(argv))


if __name__ == '__main__':
    main(sys.argv[1:])