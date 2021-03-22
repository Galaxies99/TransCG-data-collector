import re
import sys
import getopt

def parse_cmd(cmd):
    cmd_res = []
    cmd_part = ""
    s1 = False
    s2 = False
    prev = ''
    for c in cmd:
        if c == ' ' or c == '\t' or c == '\n':
            if s1 or s2:
                cmd_part = cmd_part + c
            else:
                if cmd_part != "":
                    cmd_res.append(cmd_part)
                    cmd_part = ""
            continue
        if c != '\\' or (c == '\\' and prev == '\\'):
            cmd_part = cmd_part + c
        if c == "'":
            if prev != '\\':
                s1 = not s1
        if c == '"':
            if prev != '\\':
                s2 = not s2
        prev = c

    if cmd_part != "":
        cmd_res.append(cmd_part)
    
    return cmd_res


def parse_ipv4(addr):
    addr = addr.lstrip(' ').rstrip(' ')
    addr_list = addr.split(':')
    if len(addr_list) != 2:
        return False
    ip, port = addr_list
    if port.isdigit() is False:
        return False
    port = int(port)
    if port < 0 or port > 65535:
        return False
    ip_stack = ip.split('.')
    if len(ip_stack) != 4:
        return False
    for item in ip_stack:
        if item.isdigit() is False:
            return False
        item = int(item)
        if item < 0 or item > 255:
            return False
    return True


def parse_argv_relay(argv):
    src_addr_t = ""
    try:
        opts, _ = getopt.getopt(argv, "hs:", ["help", "src="])
    except getopt.GetoptError:
        print("relay.py -s <sourceAddr> ")
        print("(or)     --src=<sourceAddr>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("relay.py -s <sourceAddr>")
            print("(or)     --src=<sourceAddr>")
            sys.exit()
        elif opt in ("-s", "--src"):
            src_addr_t = arg
    if parse_ipv4(src_addr_t) is False:
        print("[Error] sourceAddr is invalid!")
        sys.exit(2)
    src_ip, src_port = src_addr_t.split(':')
    src_port = int(src_port)
    src_addr = (src_ip, src_port)
    return src_addr


def parse_argv_client(argv):
    dst_addr_t = ""
    err = False
    try:
        opts, _ = getopt.getopt(argv, "hd:e", ["help", "dst=", "error"])
    except getopt.GetoptError:
        print("client.py -d <destinationAddr>[ -e]")
        print("(or)      --dst=<destinationAddr>[ --error]")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("client.py -d <destinationAddr>")
            print("(or)      --dst=<destinationAddr>")
            sys.exit()
        elif opt in ("-d", "--dst"):
            dst_addr_t = arg
        elif opt in ("-e", "--error"):
            err = True
    if parse_ipv4(dst_addr_t) is False:
        print("[Error] destinationAddr is invalid!")
        sys.exit(2)
    dst_ip, dst_port = dst_addr_t.split(':')
    dst_port = int(dst_port)
    dst_addr = (dst_ip, dst_port)
    return dst_addr, err
