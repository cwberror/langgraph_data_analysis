#!/usr/bin/env python3
import socket

def check_port(port):
    """检查指定端口是否被监听"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)  # 设置超时时间
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"检查端口 {port} 时出错: {e}")
        return False

# 检查MCP服务器端口
ports_to_check = [8006, 8007]
for port in ports_to_check:
    is_listening = check_port(port)
    print(f"端口 {port}: {'正在监听' if is_listening else '未监听'}")