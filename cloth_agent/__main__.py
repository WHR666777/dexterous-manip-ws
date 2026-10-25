"""统一模块入口；spawn 子进程导入时不会再次启动 CLI。"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
