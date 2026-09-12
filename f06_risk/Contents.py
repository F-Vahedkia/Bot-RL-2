"""
contracts.py (1)===/1
limits.py (2)===/2
state.py (3)===/3
risk_engine.py (4)===/4
action_builder.py (5)===/5
f06_risk_tester_A.py (6)                          ---> tester:   pytest -v -s f06_risk/f06_risk_tester_A.py
risk_context.py (7)===/6
risk_context_tester_A.py (8)                      ---> tester:   pytest -v -s f06_risk/risk_context_tester_A.py
risk_projection.py (9)===/7
risk_projection_tester_A.py (10)                  ---> tester:   pytest -v -s f06_risk/risk_projection_tester_A.py
risk_engine_projection_integration_tester_A.py (11) -> tester:   pytest -v -s f06_risk/risk_engine_projection_integration_tester_A.py
margin_calculator.py (12)===/8
margin_calculator_tester_A.py (13)                ---> tester:   pytest -v -s f06_risk/margin_calculator_tester_A.py
position_sizing.py (14)===/9
position_sizing_tester_A.py (15)                  ---> tester:   pytest -v -s f06_risk/position_sizing_tester_A.py
projection_margin.py (16)===/10
projection_margin_tester_A.py (17)                ---> tester:   pytest -v -s f06_risk/projection_margin_tester_A.py
risk_projection_margin_integration_tester_A (18)  ---> tester:   pytest -v -s f06_risk/risk_projection_margin_integration_tester_A
risk_engine_instrument_margin_tester_A.py (19)    ---> tester:   pytest -v -s f06_risk/risk_engine_instrument_margin_tester_A.py
---------------------
بخش 1
---------------------
risk_engine_portfolio_risk_boundary_tester_A.py (20) -> tester
risk_engine_position_count_boundary_tester_A.py (21) -> tester
risk_context_symbol_position_count_contract_tester_A.py (22) -> tester
risk_projection_symbol_position_count_contract_tester_A.py (23) ->tester
risk_engine_symbol_position_count_boundary_tester_A.py (24) -> tester
projection_margin_contract_tester_A.py (25) -> tester
position_sizing_contract_tester_A.py (26) -> tester
margin_calculator_contract_tester_A.py (27) ->tester

=======================================================
اجرای همزمان تمام فایلهای تستر
=======================================================
pytest -v -s `
f06_risk/f06_risk_tester_A.py `
f06_risk/risk_context_tester_A.py `
f06_risk/risk_projection_tester_A.py `
f06_risk/risk_engine_projection_integration_tester_A.py `
f06_risk/margin_calculator_tester_A.py `
f06_risk/position_sizing_tester_A.py `
f06_risk/projection_margin_tester_A.py `
f06_risk/risk_projection_margin_integration_tester_A.py `
f06_risk/risk_engine_instrument_margin_tester_A.py `
f06_risk/risk_engine_portfolio_risk_boundary_tester_A.py `
f06_risk/risk_engine_position_count_boundary_tester_A.py `
f06_risk/risk_context_symbol_position_count_contract_tester_A.py `
f06_risk/risk_projection_symbol_position_count_contract_tester_A.py `
f06_risk/risk_engine_symbol_position_count_boundary_tester_A.py `
f06_risk/projection_margin_contract_tester_A.py `
f06_risk/position_sizing_contract_tester_A.py `
f06_risk/margin_calculator_contract_tester_A.py `

================================================================================================ 1
میزان باقی‌مانده فصل ۴

به‌صورت واقع‌بینانه، فصل ۴ را الآن حدود ۷۰٪ تکمیل‌شده در نظر می‌گیرم؛ یعنی تقریباً ۳۰٪ کار اصلی باقی مانده است.

بخش بنیادی Risk Layer ساخته و با regression کامل تثبیت شده؛
قسمت باقی‌مانده بیشتر مربوط به production hardening، تکمیل قواعد ریسک و آزمون سناریوهای پیچیده است.

وضعیت فعلی Chapter 4 — Risk Layer
✅ انجام شده
فصل ۴ — Risk Layer

✅ contracts.py
✅ limits.py
✅ state.py
✅ risk_context.py
✅ risk_projection.py
✅ risk_engine.py
✅ action_builder.py

✅ PositionSizingCalculator
   12/12

✅ MarginCalculator
   12/12

✅ ProjectionMarginCalculator
   11/11

✅ RiskProjection ↔ Margin integration
   12/12 + 6/6

✅ RiskEngine ↔ RiskProjection
   12/12 + 6/6

✅ RiskEngine ↔ Instrument-aware Margin
   6/6

✅ Regression کامل Chapter 4
   88/88
وضعیت عملکردی فعلی

مسیر اصلی اکنون به‌صورت زیر کار می‌کند:

PortfolioDecision
        ↓
RiskRequest
        ↓
RiskContext
        ↓
RiskProjection
        ↓
ProjectedRiskState
        ↓
Risk Constraints
        ↓
Margin Projection
        ↓
RiskDecision

محدودیت‌های اصلی فعلی شامل:
exposure، total exposure، correlation، concentration، margin utilization، drawdown و portfolio-risk budget هستند.
مسیر projected نیز پس از اعمال modification دوباره projection را محاسبه می‌کند.

🔲 کارهای باقی‌مانده (دو تا کار اول انجام شدند).
1. ✅ تکمیل Portfolio Risk constraints ==> 7 تست پاس شد. فایل شماره بیست

هنوز باید تعریف کنیم portfolio risk دقیقاً چه چیزی را اندازه می‌گیرد و
چگونه با exposure، margin و position risk ارتباط پیدا می‌کند.

2. ✅ Position-count / per-symbol constraints

محدودیت‌هایی مثل:

maximum open positions
maximum positions per symbol
maximum simultaneous directional exposure

هنوز به‌صورت کامل در Risk Layer تثبیت نشده‌اند.

3. 🔲 Leverage / Free-Margin hard guards

باید guardهای صریح برای شرایط خطرناک مانند:

insufficient free margin
invalid margin level
excessive leverage
zero/negative usable capital

تعریف و تست شوند.

4. 🔲 Hard vs Soft constraints + Conflict Resolution

این یکی از مهم‌ترین بخش‌های باقی‌مانده است.

مثلاً اگر همزمان:

Correlation limit
        +
Margin limit
        +
Symbol concentration
        +
Total exposure

نقض شوند، باید اولویت و ترتیب حل تعارض‌ها کاملاً رسمی، deterministic و قابل audit باشد.

در کد فعلی ترتیب اعمال محدودیت‌ها وجود دارد، اما هنوز باید به‌عنوان
policy معماری به‌صورت رسمی تثبیت و با conflict scenarios تست شود.

5. 🔲 Net Exposure / Direction

باید به‌طور رسمی مشخص شود:

gross exposure
net exposure
long exposure
short exposure

و مخصوصاً رفتار موقعیت موجود + موقعیت پیشنهادی چگونه محاسبه می‌شود.

6. 🔲 Stop-Loss / Risk-per-Trade

اگر این موارد بخشی از قرارداد نهایی Risk باشند، باید به Risk Layer وارد شوند:

stop distance
risk per trade
monetary risk
portfolio risk contribution
7. 🔲 Audit / Lineage

باید مشخص شود RiskDecision چگونه ثبت می‌کند:

original decision
modified decision
violated limits
modification factors
projection state
risk-engine version

تا تصمیم قابل بازسازی و audit باشد.

8. 🔲 Mode-aware behavior

رفتار Risk Layer برای:

train
optimize
backtest
replay
eval
shadow
paper
live

باید صریحاً بررسی و تثبیت شود.

9. 🔲 Multi-symbol / Conflict Scenarios

تست‌های فعلی بسیار خوب‌اند، اما هنوز به مجموعه‌ای از سناریوهای پیچیده‌تر نیاز داریم:

Symbol cap + Correlation
Correlation + Total exposure
Total exposure + Margin
Margin + Free margin
Drawdown + Exposure
چند محدودیت همزمان
10. 🔲 End-to-End Chapter 4 Test

یک تست رسمی End-to-End لازم است که کل مسیر فصل ۴ را از:

PortfolioDecision
        ↓
RiskRequest
        ↓
RiskEngine
        ↓
RiskDecision

در یک سناریوی واقعی چندنمادی بررسی کند.

11. 🔲 Final Architecture Review

در پایان باید کل Chapter 4 بازبینی شود تا مشخص شود:

contract
responsibility
dependency
API
immutability
determinism
error handling
auditability

با معماری کل v8 سازگار هستند.

12. 🔲 بستن رسمی Chapter 4

پس از سبز شدن تست‌های نهایی:

Chapter 4 = CLOSED

و سپس می‌توانیم وارد Chapter بعدی شویم.

جمع‌بندی فعلی -----------------------------------------
Chapter 4 — Risk Layer

✅ Foundation                         COMPLETE
✅ Core contracts                     COMPLETE
✅ Risk context                       COMPLETE
✅ Projection                         COMPLETE
✅ Position sizing                    COMPLETE
✅ Margin calculation                 COMPLETE
✅ Instrument-aware margin            COMPLETE
✅ RiskEngine                         FUNCTIONAL
✅ Integration                        COMPLETE
✅ Regression                         88/88 GREEN

🔲 Advanced risk constraints
🔲 Conflict-resolution policy
🔲 Position-count / leverage guards
🔲 Net/gross/directional risk
🔲 Stop-loss / risk-per-trade
🔲 Audit & lineage
🔲 Mode behavior
🔲 Advanced multi-symbol scenarios
🔲 End-to-End test
🔲 Final architecture review
وضعیت فعلی قابل انتقال به چت بعدی

Chapter 4 حدود 70٪ کامل است.
Foundation و مسیر عملیاتی اصلی Risk Layer ساخته شده و Regression کامل 88/88 سبز است.
Instrument-aware Margin نیز در مسیر واقعی RiskEngine تست شده و 6/6 تست آن سبز است.
حدود 30٪ باقی‌مانده عمدتاً شامل:
advanced constraints، conflict resolution، leverage/free-margin guards،
net/gross risk semantics، risk-per-trade/SL در صورت نهایی‌شدن قرارداد،
audit/lineage، mode-awareness، سناریوهای چندنمادی پیچیده، End-to-End و final review/closure است.

================================================================================================ 
"""