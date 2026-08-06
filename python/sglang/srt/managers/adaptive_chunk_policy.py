from dataclasses import dataclass

@dataclass(frozen=True)
class AdaptiveChunkConfig:
    min_chunk: int = 512
    small_chunk: int = 1024
    medium_chunk: int = 2048
    max_chunk: int = 4096

    decode_low_watermark: int = 4
    decode_high_watermark: int = 12

    prefill_queue_high_watermark: int = 8

    def validate(self,
                 *,
                 page_size:int,
                 launch_chunk_size:int,
                 ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be > 0")

        chunks = (self.min_chunk, self.small_chunk, self.medium_chunk, self.max_chunk)

        if any(chunk <=0 for chunk in chunks):
            raise ValueError("All chunks must be >= 0")

        if tuple(sorted(chunks)) != tuple(chunks):
            raise ValueError("All chunks must be in ascending order")

        if self.max_chunk > launch_chunk_size:
            raise ValueError("adaptive max chunk cannot exceed the launch-time chunked_prefill_size")

        if any(chunk % page_size != 0 for chunk in chunks):
            raise ValueError("all adaptive chunk sizes must be divisible by page_size")

        if self.decode_low_watermark < 0 :
            raise ValueError("decode_low_watermark must be >= 0")

        if self.decode_high_watermark <= self.decode_low_watermark :
            raise ValueError("decode_high_watermark must be greater than decode_low_watermark")

        if self.prefill_queue_high_watermark < 0 :
            raise ValueError("prefill_queue_high_watermark must be >= 0")


class AdaptiveChunkPolicy:
    def __init__(self,
                 config: AdaptiveChunkConfig,
                 *,
                 page_size:int,
                 launch_chunk_size:int) -> None:
        config.validate(page_size=page_size, launch_chunk_size=launch_chunk_size)
        self.config = config


    def choose(self,
               *,
               decode_batch_size:int,
               waiting_prefill_size:int,
               ) -> int:
        if decode_batch_size < 0:
            raise ValueError("decode_batch_size must be >= 0")

        if waiting_prefill_size < 0:
            raise ValueError("waiting_prefill_size must be >= 0")

        cfg = self.config

        if decode_batch_size == 0:
            # 没有decode时：最大程度提升prefill性能
            chunk = cfg.max_chunk
        elif decode_batch_size <= cfg.decode_low_watermark:
            # decode压力升高，prefill chunk 减小，降低对decode的压力
            chunk = cfg.medium_chunk
        elif decode_batch_size <= cfg.decode_high_watermark:
            chunk = cfg.small_chunk
        else:
            chunk = cfg.min_chunk

        #prefill产生积压时，提高chunk等级 TODO:这里写的不太优雅，性能有欠缺
        if waiting_prefill_size >= cfg.prefill_queue_high_watermark:
            levels = (cfg.min_chunk,
                      cfg.small_chunk,
                      cfg.medium_chunk,
                      cfg.max_chunk,)
            index = levels.index(chunk)
            chunk = levels[min(index+1, len(levels) -1)]

        return chunk


