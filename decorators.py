from functools import wraps
from flask import request, jsonify
from auth import decode_token

def token_required(secret_key):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')

            if not token:
                return jsonify({'message': 'Token is missing!'}), 401

            token = token.split(" ")[1]  # Extract token from "Bearer <token>"

            decoded_token = decode_token(token, secret_key)

            if 'error' in decoded_token:
                return jsonify({'message': decoded_token['error']}), 401

            return f(*args, **kwargs)

        return decorated
    return decorator
