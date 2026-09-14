# 등검은말벌(Vespa velutina) 확산-경쟁 PDE 모델 스펙

## 1. 배경 및 목적

남한 전역을 500m × 500m 해상도 격자로 이산화한 공간 도메인에서, 외래종인 등검은말벌(*Vespa velutina*)의 확산을 시뮬레이션한다. 등검은말벌은 SDM(Species Distribution Model) 산출물을 이용해 환경수용력이 높은 방향으로 이동(taxis)하며, 토착종인 장수말벌(*Vespa mandarinia*)과 로트카-볼테라 형태로 경쟁한다.

## 2. 상태 변수 및 입력 필드

| 기호 | 의미 | 데이터 소스 / 비고 |
|---|---|---|
| $u(x,t)$ | 등검은말벌(외래종) 개체 밀도 | PDE로 시간발전 |
| $v(x)$ | 장수말벌(토착종) 필드 | 시간 독립, 정적 필드 |
| $C_u(x)$ | 등검은말벌의 위치별 환경수용력(carrying capacity) | `/raw/SAM/density.csv` — taxis의 potential이자 로지스틱 성장의 수용력으로 동시에 사용 |
| $S_v(x)$ | 장수말벌의 위치별 환경적합도 | `/raw/jangsu/suitabilty.csv` — 정규화(0~1) 가정 |

$x=(x_1,x_2)\in\Omega$, $\Omega$는 남한 landmass 형태를 딴 격자이며 셀 크기는 $\Delta x=\Delta y=500\text{m}$ (바다/영역 밖은 마스킹 필요).

> **가정 (확인 필요)**: `density.csv`가 이미 "환경수용력"(최대 서식밀도 스케일까지 반영된 값)이라고 보고, 별도의 상수 $K$ 없이 $C_u(x)$를 로지스틱 성장의 수용력으로 그대로 사용했습니다. 또한 taxis가 "따라가는" 방향도 이 동일한 필드의 기울기로 두었습니다(적합도와 환경수용력이 같은 공간패턴을 따른다고 가정). 만약 taxis용 적합도 필드와 성장용 수용력 필드가 서로 다른 데이터여야 한다면 알려주세요.

## 3. 최종 지배방정식

$$
\frac{\partial u}{\partial t} = \underbrace{D_u \nabla^2 u}_{\text{확산}} \;-\; \underbrace{\chi_u \nabla\cdot\big(u\,\nabla C_u(x)\big)}_{\text{환경수용력 taxis}} \;+\; \underbrace{\alpha\, u\left(1 - \frac{u}{C_u(x)}\right)}_{\text{로지스틱 성장 (수용력 = 환경수용력 그 자체)}} \;-\; \underbrace{\beta\, u\, v(x)}_{\text{장수말벌과의 경쟁}}
$$

$$
v(x) = K_v \, S_v(x) \qquad (\text{시간에 대해 고정된 정적 필드})
$$

### 항 별 설명

- **확산항** $D_u \nabla^2 u$: 등검은말벌의 무작위 확산(랜덤워크). $D_u$는 확산계수.
- **Taxis항** $-\chi_u \nabla\cdot(u\nabla C_u)$: Keller-Segel 형태의 이류항. 개체가 환경수용력 $C_u$의 기울기를 따라 이동. $\chi_u>0$는 taxis 민감도.
- **성장항** $\alpha u(1-u/C_u(x))$: 로지스틱 성장, 국소 수용력이 `density.csv`에서 읽어온 $C_u(x)$ 그대로.
- **경쟁항** $-\beta u v(x)$: 로트카-볼테라 형태의 편측 경쟁(one-way competition). 장수말벌 밀도가 높은 곳일수록 등검은말벌 성장이 억제됨. $v$가 $u$의 영향을 받지 않는 편도 구조.

## 4. 경계조건

- No-flux (Neumann) 경계조건: 도메인 경계(해안선 등)에서 순플럭스가 0이 되도록 설정.

$$
\left(D_u \nabla u - \chi_u\, u\, \nabla C_u\right)\cdot \mathbf{n} = 0 \quad \text{on } \partial\Omega
$$

- 남한은 직사각형이 아닌 불규칙한 landmass이므로, **land/sea 마스크**를 만들어 바다 영역은 계산에서 제외하고, 해안선(마스크 경계)에서 no-flux를 적용해야 함 (ghost-cell 또는 mirror 방식 권장).

## 5. 공간/시간 이산화

- 셀 크기: $\Delta x=\Delta y=500\text{m}$ (고정)
- 격자 행/열 개수: 500×500으로 고정된 것이 아니라, `density.csv` / `suitabilty.csv` 래스터의 실제 남한 커버 범위 ÷ 500m 로 결정됨 (두 파일의 grid shape·좌표계가 서로 일치하는지 반드시 확인)
- 시간적분: taxis항(이류적 특성) 때문에 밀도의 음수화(negative density) 문제가 생길 수 있으므로, upwind 방식 또는 Patankar-type 양수 보존 스킴 권장
- 안정성: 명시적(explicit) 시간적분을 쓸 경우 확산항과 taxis항 모두에 대한 CFL 조건 확인 필요 (taxis 계수 $\chi_u$가 크면 시간간격을 작게 잡아야 함)

## 6. 데이터 소스

| 파일 | 내용 | 사용처 |
|---|---|---|
| `/raw/SAM/density.csv` | 등검은말벌 환경수용력 $C_u(x)$ | taxis 항의 potential, 성장항의 수용력 |
| `/raw/jangsu/suitabilty.csv` | 장수말벌 환경적합도 $S_v(x)$ | $v(x)=K_v S_v(x)$ 계산 |

## 7. 아직 정해지지 않은 항목 (구현 전 확정 필요)

- [ ] 파라미터 값: $D_u, \chi_u, \alpha, \beta, K_v$
- [ ] 초기조건 $u(x,0)$: 등검은말벌의 실제 최초 유입 지점(예: 부산항 등)을 기반으로 한 국소 분포로 설정할지 여부
- [ ] `density.csv`, `suitabilty.csv`의 실제 포맷 확인 (행/열이 위경도 격자인지, 좌표계/CRS, 셀 해상도가 정말 500m인지, 두 파일의 grid가 서로 정렬되어 있는지)
- [ ] $S_v$ 정규화 범위(0~1 가정) 확인
- [ ] $C_u$가 taxis와 성장에 동일하게 쓰인다는 위 가정이 맞는지 확인
- [ ] 시뮬레이션 기간 및 시간 스텝 크기

## 8. 향후 확장 가능성 (참고)

- $v$를 정적 필드가 아닌 시간발전 필드로 확장하고 싶을 경우:
  - 완화형: $\partial v/\partial t = r_v(K_v S_v(x) - v)$
  - 완전 반응-확산형: $\partial v/\partial t = D_v \nabla^2 v + r_v v(1-v/(K_v S_v(x)))$ (+ 필요시 $-\gamma uv$ 상호경쟁항 추가)
