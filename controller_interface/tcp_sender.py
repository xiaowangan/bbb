import json
import socket
import struct
from typing import Any, Dict, Tuple


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """
    从 socket 中精确读取指定字节数。
    """
    chunks = []
    received = 0

    while received < num_bytes:
        chunk = sock.recv(num_bytes - received)
        if not chunk:
            raise ConnectionError("连接已关闭，未收到完整数据")
        chunks.append(chunk)
        received += len(chunk)

    return b"".join(chunks)


def send_json_message(
    host: str,
    port: int,
    payload: Dict[str, Any],
    timeout: float = 5.0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    发送一条 JSON 消息，并等待对方返回一条 JSON ACK。

    协议：
    - 先发 4 字节无符号大端整数，表示正文长度
    - 再发 UTF-8 JSON 正文
    - 接收端同样返回相同格式

    返回：
    - (是否成功, 对方响应dict)
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = struct.pack("!I", len(body))

    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)

        sock.sendall(header)
        sock.sendall(body)

        ack_header = _recv_exact(sock, 4)
        ack_len = struct.unpack("!I", ack_header)[0]

        ack_body = _recv_exact(sock, ack_len)
        ack_data = json.loads(ack_body.decode("utf-8"))

    success = bool(ack_data.get("ok", ack_data.get("success", False)))
    return success, ack_data


def send_trajectory_payload(
    host: str,
    port: int,
    trajectory_payload: Dict[str, Any],
    timeout: float = 5.0,
) -> None:
    """
    发送轨迹并打印结果。
    """
    ok, ack = send_json_message(
        host=host,
        port=port,
        payload=trajectory_payload,
        timeout=timeout,
    )

    print("\n" + "=" * 60)
    print("TCP发送结果")
    print("=" * 60)
    print(f"目标地址: {host}:{port}")
    print(f"发送是否成功: {ok}")
    print("对方返回:")
    print(json.dumps(ack, ensure_ascii=False, indent=2))
    print("=" * 60)