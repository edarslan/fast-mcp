import json
from mcp.server.fastmcp import FastMCP

# --------- Simple Prompt Manager ---------
class PromptManager:
    def __init__(self):
        # Two example templates
        self.prompts = {
            'greet': 'Hello, {name}! Welcome to our MCP server.',
            'sum_prompt': 'Calculate the sum of {a} and {b}.'
        }

    def render(self, prompt_id, **kwargs):
        template = self.prompts.get(prompt_id)
        if not template:
            return f"Prompt '{prompt_id}' not found."
        try:
            return template.format(**kwargs)
        except KeyError as e:
            return f"Missing variable {e} for prompt '{prompt_id}'."

# --------- Simple MCP Server ---------
class SimpleMCPServer:
    def __init__(self):
        # Create FastMCP instance
        self.mcp = FastMCP('SimpleMCP')
        self.prompt_manager = PromptManager()
        # Register tools and resources
        self._register_tools()
        self._register_resources()

    def _register_tools(self):
        @self.mcp.tool()
        def echo(text: str) -> str:
            """Return the same text back to the user."""
            return text

        @self.mcp.tool()
        def add(a: int, b: int) -> int:
            """Return the sum of two numbers."""
            return a + b

        @self.mcp.tool()
        def render_prompt(prompt_id: str, **kwargs) -> str:
            """Render a sample prompt with given variables."""
            return self.prompt_manager.render(prompt_id, **kwargs)

    def _register_resources(self):
        @self.mcp.resource('config://info')
        def config_info() -> str:
            """Provide basic server configuration information."""
            info = {
                'server_name': self.mcp.name,
                'supported_tools': list(self.mcp._tools.keys()),
                'supported_prompts': list(self.prompt_manager.prompts.keys()),
            }
            return json.dumps(info, indent=2)

        @self.mcp.resource('tools://list')
        def tools_list() -> str:
            """List available tools."""
            return json.dumps(list(self.mcp._tools.keys()), indent=2)

    def run(self, transport='stdio'):
        print(f"Starting SimpleMCPServer on transport '{transport}'...")
        self.mcp.run(transport=transport)

# --------- Entry Point ---------
if __name__ == '__main__':
    server = SimpleMCPServer()
    server.run()
