class BridgeError(Exception):
    def __init__(self, status: int, code: int, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message

    def body(self):
        return {"error_code": self.code, "error_msg": self.message}


class ConfigError(ValueError):
    pass


def invalid(message):
    return BridgeError(400, 2002, message)
