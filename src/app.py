import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chain import CustomerServiceChain


WELCOME = """
╔══════════════════════════════════════════════════╗
║       智能问答客服系统 - AI Customer Service        ║
║      知识库: 产品手册 | 退换货政策 | 客服话术     ║
╚══════════════════════════════════════════════════╝

输入您的问题开始对话，输入 /help 查看命令
"""

HELP_TEXT = """
命令列表:
  /help     - 显示此帮助信息
  /clear    - 清除对话记忆
  /exit     - 退出系统

客服热线: 400-XXX-XXXX
"""


def main():
    print(WELCOME)

    try:
        chain = CustomerServiceChain()
    except RuntimeError as e:
        print(f"\n[启动失败] {e}")
        sys.exit(1)

    print("\n" + "─" * 50)
    print(HELP_TEXT)
    print("─" * 50 + "\n")

    while True:
        try:
            user_input = input("您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/exit":
                print("再见！")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
                continue
            elif cmd == "/clear":
                chain.clear_memory()
                continue
            else:
                print(f"未知命令: {user_input}，输入 /help 查看可用命令")
                continue

        print("\n" + chain.answer(user_input) + "\n")


if __name__ == "__main__":
    main()
