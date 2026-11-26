# RFF-2.0 — разбор алгоритмов как референса для deep zoom

Репозиторий: `~/RFF-2.0` (Merutilm/RFF-2.0). Это **Ridiculously Fast Fractal** —
C++ / Vulkan рендер с упором на скорость. Использовать его как продукт нельзя
(Vulkan + GMP + OpenCV + GLFW, свои форматы, `-ffast-math`, compute shader в
`float` и честное предупреждение в README про неточность). Но алгоритмы —
именно тот класс, который нам нужен для 10^16 итераций.

Наши текущие средства: BLA 2-го порядка (`bla.c`), MPFR-центр с динамической
точностью (`orbit_precision()`), потолок 2^25, полная орбита `2×(N+1)` doubles.
RFF решает те же проблемы иначе: **никогда не хранит и не проходит N итераций
по пикселю**.

---

## 1. Карта исходников

| Файл | Роль |
|---|---|
| `algorithm/MB2ReferenceGenerator.hpp` | генерация опорной орбиты, FPG, period-кандидаты, чекпоинты, сжатие |
| `algorithm/ReferenceCheckpoint.hpp` | чекпоинт: `fixed_point_complex z`, `refIteration`, `fzgAn` |
| `calc/fixed_point_complex.hpp`, `fixed_point_decimal.hpp` | произвольная точность на GMP-limb'ах, `exp10` |
| `algorithm/PA.h` | примитив skip: `skip, an, bn, radius` |
| `algorithm/PAGenerator.h` | рекуррентное построение `an/bn/radius` по орбите, `merge` |
| `algorithm/MPAInfo.cpp/.hpp` | иерархия периодов, искусственные периоды, счётчики |
| `algorithm/MB2MPAGenerator.hpp` | построение таблицы MPA (uncompressed/compressed) |
| `algorithm/MPAView.hpp` | runtime-выбор PA для заданных `refIteration` и `dz` |
| `algorithm/MPAUtils.hpp`, `MPAIndexMapperUtils.hpp` | индексация плоской таблицы |
| `algorithm/srpc_module/*` | сжатие повторов reference и MPA-таблицы (SRPC) |
| `algorithm/MB2Perturbator.h` | perturбация + MPA + SA + interior-тесты + SIMD |
| `data/ReferenceData.hpp`, `data/MPAData.hpp`, `data/ApproxTableBuffer.h` | результаты и буферы |

Счётчик итераций — `itt::ptb = uint128_t` (`calc/iteration_types.hpp`), формально
до 2^128.

---

## 2. Опорная орбита: FPG + record-minima + чекпоинты

`MB2ReferenceGenerator::generate()` (`MB2ReferenceGenerator.hpp:237`):

```cpp
for (refIteration = 0; tryUpdateAnBn(radius2) && z0.norm_sqr() < bailoutSqr; ++refIteration) {
    tryAppendPeriodCandidate(minZRadius, radius2);      // минимум |z| → период
    if (refIteration % refSettings.partitionSize == 0)
        placeCheckpoint();                               // z + fzgAnPartition
    fn(refIteration);
    stepOnceOptimal(prevExp2div64, specialCenterType);   // z ← z² + c
    processor.put(z0);                                   // сжатое хранение z0
}
```

Ключевые детали:

- **FPG (Fast Period Guessing).** `tryUpdateAnBn()` (`134–148`) ведёт
  `fzgAn = Π(2·z_k)` и `fpgBn = 2·z0·fpgBn + 1` с проверкой
  `|fpgBn_temp|·dcMax > |z0|²` — это радиус применимости линейной
  аппроксимации. `try_normalized_value()` удерживает экспоненту, чтобы не
  было переполнения.
- **Период-кандидат** — индекс, на котором `|z|` обновляет рекордный минимум
  (`tryAppendPeriodCandidate`, `155–160`). Возле миниброта орбита
  регулярно возвращается близко к предыдущему состоянию — эти индексы и есть
  длины скипов.
- **Чекпоинт** каждые `partitionSize` итераций: `{z (высокая точность),
  refIteration, fzgAnPartition}` — ровно то, что я предлагал как этап 1.
- **`syncReference()` (`102–118`)** — periodic re-sync `z0` из точного `z`,
  либо по интервалу, либо когда `|z0|` мал. Это и есть «rebasing» в миниатюре.
- **`updatePrecision()` (`120–132`)** — динамическая точность fixed-point
  в зависимости от экспоненты коэффициентов. Наш аналог — `orbit_precision()`.
- **SRPC/ComplexApproximator** (`ComplexApproximator.hpp:16`) считает
  повтор, если `|b/a − 1| <= threshold` для Re и Im: дубликаты орбиты
  заменяются ссылками на range.

Итог:-reference хранится **сжатой** (SRPC), а не `2N` doubles, и чекпоинтов
всего `threads+1`.

---

## 3. PA — первый порядок вместо нашего BLA

`PA<Num>` (`PA.h:11`):

```cpp
itt::ref skip;
complex<Num> an, bn;
Num radius;

complex<Num> apply(dz, dc) const { return an*dz + bn*dc; }
bool isValid(dzRad) const { return dzRad < radius; }
```

Это **линейное** возмущение: `dz_{k+n} ≈ an·dz_k + bn·dc`. Наш BLA — второй
порядок с кубическими членами; RFF сознательно берёт первый, но зато делает
иерархию периодов и тем компенсирует длину скипа.

`PAGenerator` (`PAGenerator.h:98`):

```cpp
z2     = 2 * Z[start + skip++];
radius = min(radius, clamp((epsilon*|z2| - |bn|*dcMax) / |an|, 0, radius));
an     = (an * z2).normalize();
bn     = (bn * z2 + 1).normalize();
```

Интерпретация: `an` — произведение `2Z_k`, `bn` — накопленная производная по
`dc`, `radius` — гарантированная область, где отброшенные члены меньше
`epsilon`. `dcMax` — максимальный `|dc|` в тайле. Всё normalize-ится по
экспоненте — наш `struct h` из `bla.c` делает то же самое.

Композиция (`merge`, `78–95`):

```cpp
skip  += target.skip;
radius = clamp((target.radius - |bn|*dcMax)/|an|, 0, radius);
an     = target.an * an;
bn     = target.an * bn + target.bn;
```

Это и есть «Multi-Periodic Approximation»: склеиваем skipping разных периодов.

---

## 4. Иерархия периодов MPA

`MPAInfo::generateTablePeriod()` (`MPAInfo.cpp:72`) берёт list period-кандидатов
из reference и **вставляет искусственные периоды** так, чтобы соседние уровни
отличались не больше чем в `maxMultiplierBetweenLevel` раз:

```cpp
while (current * mult * mult < p) {
    tablePeriods.push_back(current * mult);   // искусственный период
    current *= mult;
}
tablePeriods.push_back(p);                     // настоящий период
```

Зачем: если периоды растут экспоненциально, число уровней ~ `log(N)`, а не N.
Даже для 10^16 таблица остаётся маленькой. `isArtificial[]` помечает такие
уровни.

`generateCountInfos()` (`25–70`) делает mixed-radix разложение: для каждого
уровня считает `skippableIterationCounts[i]` сколько итераций покрывает
уровень i) и `tableElementCounts[i]` (сколько PA-записей в таблице). Пример из
комментариев: периоды `[3, 11, 26]` → skippable `[1, 3, 7, 21]`, elements
`[1, 4, 10, 30]`.

`MPAIndexMapper` связывает `iteration → {mapped offset, generatedLevels}` —
плоский буфер PA-записей, по одной на уровень на каждую итерацию.

Сжатый путь (`pulledMPACompressor`) выбрасывает повторяющиеся участки
таблицы — снова SRPC.

---

## 5. Построение таблицы

`MB2MPAGenerator` (`MB2MPAGenerator.hpp:136–737`):

- `fitBufferSize()` — считает размер таблицы по `tableElementCounts`, с учётом
  сжатия.
- `UProcessor/CProcessor` (`32–56`) — по уровню: `itCount`, `itCountLim`,
  `currentPA[PAGenerator]`, `currentPASkips`, `generationAvailable`, плюс
  `partialPA` для «разрезанных» композиций.
- `uStepOnce()` (`421`): пишет mapper для текущей итерации, делает
  `currentPA[0].step()`, инкрементирует счётчики.
- `uRefreshCounter()` (`393`): когда счётчик уровня дошёл до лимита —
  `merge` в следующий уровень, сброс, пересчёт лимитов. Amortized O(1).
- `uVerifyPA()` (`278`): закрытый период → записать готовую `PA` в таблицу,
  поднять level, с propagate merge.
- `cTryJumpTableGeneration()` (`212`): сжатый путь, прыжок по SRPC-
  compressor с копированием PA верхних уровней.
- `uGluePartialPA()` / `uConfigurePartialPA()` — склейка partial PA на
  границах сжатых участков.
- `PERTURBATION_REQ = 2` (`MPAUtils.hpp:16`) — на границе периода оставляются
  2 итерации perturbation, чтобы_PA не перескакивала через неё.

Сложность построения — O(длина reference × число уровней), но число уровней
логарифмическое.

---

## 6. Runtime

`MPAView::uLookup()` (`MPAView.hpp:148`):

```cpp
mapper = iteration → {offset, levels};
table  = PA[offset .. offset+levels];
r = |dz|;
pa = &table.front();
if (pa->isValid(r))
    for (j = levels; j > 1; --j)
        if (table[j-1].isValid(r)) { pa = &table[j-1]; break; }
referenceView.setIndex(refIteration + pa->skip);
return pa;
```

`MB2Perturbator::applyMPApproximation()` (`MB2Perturbator.h:102`):

```cpp
dz = pa.apply(dz, dc);
iteration    += pa.skip;
refIteration += pa.skip;
```

Дальше — обычная perturbation, плюс:

- `checkInteriorBasic()` — быстрый тест(cardioid/bulb);
- `checkInteriorAndApply()` — цикл-детектор по истории 8 last `dz`;
- `applySeriesApproximation()` — отдельный SA с `terms[]` и
  `skippedIterations` (наш будущий этап 2);
- SIMD-батчи по 16 пикселей.

---

## 7. Почему это работает на 10^16

Не потому что быстрая орбита, а потому что:

1. **N итераций не проходится.** Таблица periods + PA позволяет прыгнуть
   `skip` за один шаг; число шагов ~ число уровней × число period-кандидатов.
2. **Память не растёт как N.** Чекпоинтов `threads+1`, reference сжат SRPC,
   таблица MPA ~ `O(levels × periods)`.
3. **Экспоненциальные периоды.** Искусственные уровни `p·mult` делают
   отношение соседних уровней ограниченным → таблица остаётся компактной при
   любом N (uint128).
4. **RadiusGuard.** `PA::isValid(|dz|)` и `epsilon`-bound из `PAGenerator`
   дают сочетание скорости и контроля ошибки.

---

## 8. Что и как портировать к нам

| RFF | Наш аналог / действие |
|---|---|
| `ReferenceCheckpoint` + `syncReference` | этап 1: `orbit_at(j)` по чекпоинтам; убрать 16 байт/итерацию |
| `tryAppendPeriodCandidate` (record-min) | добавить period-массив в `compute_orbit()` |
| `PA` + `PAGenerator` (1-й порядок) | заменить/дополнить наш BLA 2-го порядка |
| `MPAInfo` (искусственные периоды) | новая иерархия level → skip |
| `MB2MPAGenerator` + `MPAView` | таблица скипов + runtime lookup в рендерере |
| `srpc_module` | сжатие длинных reference |
| `uint128` счётчик | поднять потолок ещё выше после чекпоинтов |

** stages:**

1. Checkpointed orbit (`orbit_at(j)`), чекпоинты раз в K, пересчёт ≤K шагов.
   Регрессия: текущие image должны остаться бит-идентичными.
2. FPG period detection — record-min `|z|²` в `compute_orbit()`.
3. PA 1-го порядка с `radius`, `dcMax`, normalize (`struct h`).
4. MPA-иерархия + таблица + mapper + runtime lookup вместо `bla_try_skip`.
5. SRPC сжатие.
6. Опционально оставить BLA 2-го порядка как «локальный» skip поверх MPA.

**Обязательно сохраняем:** MPFR-центр, `orbit_precision()`, SSAA=2, SDL
display, video pipeline, `--reverse-video`, pixel-identical тесты.

---

## 9. Подводные камни

- RFF использует fixed-point + `exp10`, мы — MPFR. Переносить надо math, а не
  representation.
- `epsilon` в RFF — параметр точности skip; у нас его нет, надо ввести и
  validation на Known-good сценах.
- FPG — эвристика (record minima). Возможны пропущенные периоды → нужен
  fallback на обычную perturbation.
- RFF README честно: `-ffast-math` иногда неверная картинка; compute shader в
  `float` неточный. Наш порт не должен копировать эти компромиссы.
- `PERTURBATION_REQ=2` — важно: нельзя skip-ом перепрыгнуть границу периода.
