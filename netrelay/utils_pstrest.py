import pycurl
from .utils import parse_cmd

def parse_curl(cmd):
    name, args, cmd_type = [], [], -1
    cmd_list = parse_cmd(cmd)    
    if len(cmd_list) == 0 or cmd_list[0] != 'curl':
        return name, args, cmd_type
    name.append(pycurl.URL)
    args.append(cmd_list[-1])
    cmd_list = cmd_list[1:-1]
    omit_next = False
    for i, arg in enumerate(cmd_list):
        if omit_next:
            omit_next = False
            continue
        if arg == '--header' or arg == '-H':
            name.append(pycurl.HTTPHEADER)
            if i + 1 < len(cmd_list):
                t = cmd_list[i + 1]
                if (t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'"):
                    t = t[1:-1]
                args.append(t)
            else:
                args.append("")
            omit_next = True
        if arg == '-X':
            # only support GET
            if i + 1 < len(cmd_list):
                t = cmd_list[i + 1]
                if t == 'GET':
                    cmd_type = 0
            omit_next = True
        if arg == '--request':
            # only support POST
            if i + 1 < len(cmd_list):
                t = cmd_list[i + 1]
                if t == 'POST':
                    cmd_type = 1
            omit_next = True
        if arg == '--data':
            name.append(pycurl.POSTFIELDS)
            if i + 1 < len(cmd_list):
                t = cmd_list[i + 1]
                if (t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'"):
                    t = t[1:-1]
                args.append(t)
            else:
                args.append("")
    name, args = header_combine(name, args)
    return name, args, cmd_type


def header_combine(name, args):
    res_name, res_args = [], []
    header_args = []
    for i in range(len(name)):
        if name[i] == pycurl.HTTPHEADER:
            header_args.append(args[i])
        else:
            res_name.append(name[i])
            res_args.append(args[i])
    if header_args != []:
        res_name.append(pycurl.HTTPHEADER)
        res_args.append(header_args)
    return res_name, res_args


def postprocessing_data(res):
    pos = res.find('data: {')
    if pos == -1:
        raise ValueError('Not a data record!')
    res = res[pos:]
    data = ""
    prev = ""
    first = True
    cnt = 0
    for c in res:
        if cnt == 0 and not first:
            break
        data = data + c
        if c == '{':
            if first:
                first = False
            if prev != '\\':
                cnt += 1
        if c == '}':
            if prev != '\\':
                cnt -= 1
        prev = c
    if cnt != 0:
        raise ValueError('Not a full data record!')
    return data
