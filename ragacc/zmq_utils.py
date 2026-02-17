import zmq
import zmq.asyncio
import pickle

async def async_send_recv(address, message, byte_mode=False):
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(address)

    message_to_send = pickle.dumps(message) if not byte_mode else message
    await socket.send(message_to_send)
    reply = await socket.recv()
    if not byte_mode:
        reply = pickle.loads(reply)

    socket.close()
    return reply
