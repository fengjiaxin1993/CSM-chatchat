import warnings
# 屏蔽 pydantic 模块下所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic.*")
import click
from startup import main as startup_main
from init_database import main as kb_main, create_tables
from settings import Settings
from utils import build_logger

logger = build_logger()


@click.group(help="命令行工具")
def main():
    ...


@main.command("init", help="项目初始化")
def init():
    Settings.set_auto_reload(False)
    logger.info(f"开始初始化项目数据目录：{Settings.CHATCHAT_ROOT}")
    Settings.basic_settings.make_dirs()
    logger.info("创建所有数据目录：成功。")
    create_tables()
    logger.info("初始化知识库数据库：成功。")

    Settings.create_all_templates()
    Settings.set_auto_reload(True)

    logger.info("生成默认配置文件：成功。")
    logger.warning("<red>请先检查 model_settings.yaml 里模型平台、LLM模型和Embed模型信息正确</red>")

    logger.warning("执行 kb -r 初始化知识库，然后 start -a 启动服务。")



main.add_command(startup_main, "start")
main.add_command(kb_main, "kb")

if __name__ == "__main__":
    main()
