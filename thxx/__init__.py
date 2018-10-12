__version__ = "0.1.0"


class Backend:
    def __getattr__(self, name):
        def wrap(*args):
            import torch
            for a in args:
                if torch.is_tensor(a):
                    if a.is_cuda:
                        import thxx_backend_cuda as B
                    else:
                        import thxx_backend_cpu as B
                    return getattr(B, name)(*args)
        return wrap


backend = Backend()
