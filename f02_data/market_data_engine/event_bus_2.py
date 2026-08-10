# 1
# f02_data/market_data_engine/event_bus_2.py
"""
نهایی هستند:
   - API عمومی
   - معماری Pub/Sub
   - یک Queue برای هر Consumer
   - Thread-safe
   - publish()
   - subscribe()
   - unsubscribe()

هنوز نهایی نشده اند:
   - event payload schema
   - overflow policy
   - shutdown
   - health monitoring
   - backpressure
   - metrics
   - consumer naming
   - event persistence
"""
from __future__ import annotations

from queue import Queue, Full, Empty
from threading import Lock
from typing import Dict, Any, Optional
import logging
import uuid

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class EventBus:
    """
    Thread-safe Pub/Sub EventBus.
    هر subscriber یک queue اختصاصی دریافت می‌کند.
    Producer:
        publish(...)
    Consumer:
        queue = subscribe(...)
        event = queue.get(...)
    """
    # --------------------------------------------------------- OK
    def __init__(self, queue_size: int = 1000) -> None:
        """
        در ابتدای ساخت یک شیئ از این کلاس، این موارد معلوم یا ساخته میشود:
            - اندازه طول هر صف
            - دیشکنری خالی برای مصرف کنندگان، که شماره اشتراک را میگیرد و صف را برمیگرداند
            - یک کلید، برای جلوگیری از ایجاد تداخل ها در مواقع ضروری
        """
        self._queue_size = queue_size

        # subscriber_id -> Queue
        self._subscribers: Dict[str, Queue] = {}
        self._lock = Lock()

    # --------------------------------------------------------- OK
    def subscribe_old1(self) -> str:
        """
        هربار که این متد فراخوانی شود، در دیکشنری مصرف کنندگان:
            - key: یک آیدی رشته ای به عنوان کد مصرف کننده جدید تولید میکند
            - value: یک صف اختصاصی برای آن مصرف کننده ایجاد میکند

        Create dedicated queue for consumer.
        یعنی: ایجاد صف اختصاصی برای مصرف کننده.
        Returns:
            subscriber_id
        """
        subscriber_id = str(uuid.uuid4()) 
        with self._lock:
            self._subscribers[subscriber_id] = Queue(maxsize=self._queue_size)
        logger.info("EventBus subscriber registered: %s", subscriber_id)
        return subscriber_id
    # -------------
    def subscribe(self, symbol: str) -> str:
        """
        ثبت‌نام برای دریافت رویدادهای یک نماد خاص.
        اگر symbol = "all" باشد، همه رویدادها را دریافت می‌کند.
        """
        with self._lock:
            if symbol not in self._subscribers:
                self._subscribers[symbol] = Queue(maxsize=self._queue_size)
            else:
                logger.warning(f"Symbol '{symbol}' already subscribed.")
        return symbol  # خود symbol به عنوان شناسه برمی‌گردد
    

    # --------------------------------------------------------- OK
    def unsubscribe_old1(self, subscriber_id: str) -> None:
        """
        این متد، آیدی مصرف کننده را از دیکشنری مصرف کنندگان حذف میکند.
        """
        with self._lock:
            # pop: متدی است برای حذف یک کلید از دیکشنری و دریافت مقدار مربوط به آن
            self._subscribers.pop(subscriber_id, None)
        logger.info("EventBus subscriber removed: %s", subscriber_id)
    # -------------
    def unsubscribe(self, symbol: str) -> None:
        with self._lock:
            self._subscribers.pop(symbol, None)
        logger.info("EventBus subscriber removed: %s", symbol)

        
    # --------------------------------------------------------- OK
    def get_queue_old1(self, subscriber_id: str) -> Optional[Queue]:
        """
        این متد، صف مربوط به یک مصرف کننده را برمی گرداند
        """
        with self._lock:
            return self._subscribers.get(subscriber_id)
    # -------------
    def get_queue(self, symbol: str) -> Optional[Queue]:
        """
        این متد، صف مربوط به یک مصرف کننده را برمی گرداند
        """
        with self._lock:
            return self._subscribers.get(symbol)
        

    # --------------------------------------------------------- OK
    def publish2_old1(self, event_type: str, symbol: str, timeframe: str, all_dfs: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        این متد، بدون صبر کردن، -دیکشنری- رویداد را در صف های همه مصرف کنندگان قرار می دهد
        """
        event = {
            "event_type": event_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "all_dfs": all_dfs
        }
        # ✅ ایمن - توسط ویت، ابتدا یک کپی از وضعیت فعلی می‌گیریم
        with self._lock:
            """ Learning:
            items(): یک متد است که همه جفت‌های کلید-مقدار یک دیکشنری را برمی‌گرداند
            خروجی این متد، یک view زنده است و اگر دیکشنری تغییر کند، آن خروجی هم تغییر میکند.
            گرفتن list از سطر زیر، سبب ساختن یک کپی مستقل از آن میشود.
            dict.keys()  : فقط کلیدهای دیکشنری را برمی گرداند
            dict.values(): فقط مقدارهای دیکشنری را برمی گرداند
            dict.items(): جفت های کلید-مقدار دیکشنری را برمی گرداند
            """
            subscribers = list(self._subscribers.items())

        # ✅ حالا می‌توانیم بدون قفل، روی لیست حلقه بزنیم
        for subscriber_id, q in subscribers:
            try:
                q.put_nowait(event) # put_nowait: متدی ،برای قرار دادن یک آیتم در صف، بدون منتظر ماندن است
                # logger.info("put event dict in queue")
            except Full:            # Full: استثنائی است که وقتی صف پُر باشد، توسط متد put_nowait() تولید میشود
                logger.warning(
                    "Subscriber queue full. "
                    "Dropping event. "
                    "subscriber=%s "
                    "event=%s",
                    subscriber_id,
                    event_type,
                )
    # -------------
    def publish2(self, event_type: str, symbol: str, timeframe: str, all_dfs: Dict[str, Dict[str, Any]]) -> None:
        event = {"event_type": event_type, "symbol": symbol, "timeframe": timeframe, "all_dfs": all_dfs}
        with self._lock:
            # ارسال به همان نماد
            q = self._subscribers.get(symbol)
            if q:
                try:
                    q.put_nowait(event)
                except Full:
                    logger.warning("Queue full for %s. Dropping event.", symbol)


    # --------------------------------------------------------- OK
    def get_event_old1(self, subscriber_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        این متد، از صف یک مصرف کننده معلوم، یک -دیکشنری- رویداد را دریافت میکند
        """
        q = self._subscribers.get(subscriber_id) # q: مخفف است برای queue
        if q is None:
            return None
        try:
            return q.get(timeout=timeout)
        except Empty:  # Empty: استثنائی است که وقتی تایم اوت تمام شد، توسط متد get() تولید میشود
            return None
    # -------------
    def get_event(self, symbol: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        q = self.get_queue(symbol)
        if q is None:
            return None
        try:
            return q.get(timeout=timeout)
        except Empty:
            return None

    
    # --------------------------------------------------------- OK
    def subscriber_count(self) -> int:
        """
        تعداد مصرف کنندگان را برمیگرداند
        """
        with self._lock:
            return len(self._subscribers)

    # --------------------------------------------------------- OK
    def stats(self) -> Dict[str, int]:
        """
        یک دیکشنری برمیگرداند که کلیدها، شماره های مصرف کنندگان است و
        مقادیر، برابر است با تعداد رویدادهای موجود در صف آن مصرف کننده
        """
        with self._lock:
            return {              # sid: مخفف شده است برای subscriber_id
                sid: q.qsize()    # qsize(): متدی برای گرفتن تعداد آیتم‌های موجود در صف، در همان لحظه است
                for sid, q in self._subscribers.items()
            }

    # --------------------------------------------------------- END

'''
    def publish_old(
        self,
        event_type: str,
        payload: Dict[str, Any]
    ) -> None:
    
        """
        این متد، بدون صبر کردن، -دیکشنری- رویداد را در صف های همه مصرف کنندگان قرار می دهد
        """
        event = {
            "event_type": event_type,
            "payload": payload,
        }
        # ✅ ایمن - توسط ویت، ابتدا یک کپی از وضعیت فعلی می‌گیریم
        with self._lock:
            """ Learning:
            items() یک متد است که همه جفت‌های کلید-مقدار یک دیکشنری را برمی‌گرداند
            خروجی این متد، یک view زنده است و اگر دیکشنری تغییر کند، آن خروجی هم تغییر میکند.
            گرفتن list از سطر زیر، سبب ساختن یک کپی مستقل از آن میشود.
            dict.keys()  : فقط کلیدهای دیکشنری را برمی گرداند
            dict.values(): فقط مقدارهای دیکشنری را برمی گرداند
            dict.items(): جفت های کلید-مقدار دیکشنری را برمی گرداند
            """
            subscribers = list(self._subscribers.items())

        # ✅ حالا می‌توانیم بدون قفل، روی لیست حلقه بزنیم
        for subscriber_id, q in subscribers:
            try:
                q.put_nowait(event) # put_nowait: متدی ،برای قرار دادن یک آیتم در صف، بدون منتظر ماندن است
            except Full:            # Full: استثنائی است که وقتی صف پُر باشد، توسط متد put_nowait() تولید میشود
                logger.warning(
                    "Subscriber queue full. "
                    "Dropping event. "
                    "subscriber=%s "
                    "event=%s",
                    subscriber_id,
                    event_type,
                )
'''