#!/usr/bin/env python
"""Quick test of Mistral via vLLM remote endpoint."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.app.settings import get_settings
from backend.app.llm import get_llm

print("\n" + "=" * 70)
print("Testing Mistral LLM (via vLLM remote endpoint)")
print("=" * 70)

try:
    # Load settings
    settings = get_settings()
    print(f"\n✅ Settings loaded:")
    print(f"   LLM Provider: {settings.llm_provider}")
    print(f"   LLM Model: {settings.llm_model}")
    print(f"   Remote URL: {settings.remote_llm_url}")

    # Get LLM client
    llm = get_llm()
    print(f"\n✅ LLM client initialized: {llm.__class__.__name__}")

    # Test a simple completion
    print(f"\n📝 Sending test query to Mistral...")

    response = llm.complete(
        system="You are a helpful financial assistant.",
        messages=[
            {
                "role": "user",
                "content": "What is NVIDIA known for? Answer in 1-2 sentences.",
            }
        ],
        max_tokens=100,
        temperature=0.7,
    )

    print(f"\n✅ Response from Mistral:")
    print(f"   {response}\n")

    print("=" * 70)
    print("✅ SUCCESS! Mistral is working!")
    print("=" * 70 + "\n")

except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}")
    print(f"   {str(e)}\n")
    print("=" * 70)
    print("Troubleshooting:")
    print("1. Check .env file exists with REMOTE_LLM_URL")
    print("2. Verify vLLM server is running on PSC Bridges-2")
    print("3. Check port forwarding: ssh -L 8000:v002:8000 bridges2")
    print("=" * 70 + "\n")
    sys.exit(1)
