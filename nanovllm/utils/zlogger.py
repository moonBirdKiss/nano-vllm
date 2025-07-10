from loguru import logger
import sys

logger.remove()   # 移除默认配置
logger.remove()  # 移除默认 sink

# logger.add(
#     sys.stderr,
#     level="INFO",
#     format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
#            "<level>{level: <8}</level> | "
#            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
#            "<level>{message}</level>"
# )

logger.add(
    sys.stderr,
    level="INFO",
    format="<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>"
)


logger.debug("debug visible now")
logger.info("info visible now")

__all__ = ["logger"]