# Jupyter Lab configuration for development container
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.allow_remote_access = True

# Enable extensions
c.ServerApp.jpserver_extensions = {
    'jupyter_lsp': True,
    'jupyterlab': True,
}

# Matplotlib backend
c.InlineBackend.figure_format = 'retina'
