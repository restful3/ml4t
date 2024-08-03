## 데이터베이스와 테이블 만들기

create database shop;

use shop;

CREATE TABLE goods
(
    goods_id CHAR(4) NOT NULL COMMENT '상품 ID, 고유 식별자',
    goods_name VARCHAR(100) NOT NULL COMMENT '상품 이름',
    goods_classify VARCHAR(32) NOT NULL COMMENT '상품 분류',
    sell_price INT COMMENT '판매 가격',
    buy_price INT COMMENT '구매 가격',
    register_date DATE COMMENT '등록 날짜',
    PRIMARY KEY (goods_id)
) COMMENT='상품 정보를 저장하는 테이블';

-- 모든 열에 integer나 char 등의 데이터 형식을 지정해 주어야 한다. 
-- MySQL에서 사용할 수 있는 데이터 타입의 종류는 크게 CHAR, BINARY, TEXT, VARCHAR, BLOB, 숫자형 데이터 타입이 있으며, 
-- 입력되는 데이터에 맞는 타입을 설정한다. 또한 각종 제약을 설정해줄 수 있다. 
-- null이란 데이터가 없음을 의미하며, not null은 반드시 데이터가 존재해야 한다는 의미이다. 
-- 마지막으로 goods_id 열을 기본 키(primary key)로 지정해준다.

### 테이블 정의 변경하기

-- shop 테이블에 goods_name이라는 열을 추가하며, 데이터 타입은 varchar(100)으로 설정하는 쿼리는 다음과 같다.
alter table goods add column goods_name_eng varchar(100);

-- goods_name_eng 열을 삭제하도록 한다
alter table goods drop column goods_name_eng;

### 테이블에 데이터 등록하기
insert into goods values ('0001', '티셔츠', '의류', 1000, 500, '2020-09-20');
insert into goods values ('0002', '펀칭기', '사무용품', 500, 320, '2020-09-11');
insert into goods values ('0003', '와이셔츠', '의류', 4000, 2800, NULL);
insert into goods values ('0004', '식칼', '주방용품', 3000, 2800, '2020-09-20');
insert into goods values ('0005', '압력솥', '주방용품', 6800, 5000, '2020-01-15');
insert into goods values ('0006', '포크', '주방용품', 500, NULL, '2020-09-20');
insert into goods values ('0007', '도마', '주방용품', 880, 790, '2020-04-28');
insert into goods values ('0008', '볼펜', '사무용품', 100, NULL, '2020-11-11');

## SQL 기초구문 익히기

### `select`: 열 선택하기

-- goods 테이블 중 goods_id, goods_name, buy_price 열만 선택해보도록 하자.
select goods_id, goods_name, buy_price from goods;

-- 모든 데이터를 한번에 보고 싶다면 `select * from <테이블명>;` 형태로 입력하면 된다.
SELECT * from goods;

-- goods_id, goods_name, buy_price의 이름을 바꾼 후 출력해보도록 하자.
select goods_id as id,
	goods_name as name,
	buy_price as price
from goods;

-- select 구를 통해 단순히 현재 있는 열을 선택할 뿐만 아니라 상수 및 계산식도 작성이 가능하다. 아래 쿼리를 실행해보도록 하자.
select '상품' as category,
    38 as num,
    '2022-01-01' as date,
    goods_id,
    goods_name,
    sell_price, buy_price, sell_price - buy_price as profit
from goods;

### `distinct`: 중복 제거하기
-- 중복된 데이터가 있는 경우 중복되는 값을 제거하고 고유한 값만 확인하고 싶을 때는 distinct 키워드를 사용하며, 사용법은 다음과 같다.
-- 상품 분류에 해당하는 goods_classify 열에는 중복된 값들이 존재한다. 만일 상품 분류가 어떤 것이 있는지 고유한 값만을 확인하고 싶을 경우 아래 쿼리를 실행하면 된다.
select distinct goods_classify
from goods;

### `where`: 원하는 행 선택하기
-- 여러 데이터 중 조건에 부합하는 행만 선택할 때는 where 구를 사용하면 된다. 이는 엑셀에서 필터 기능과도 비슷하다. where 구는 from 구 바로 뒤에 사용해야 작동한다.
-- 테이블에서 상품 분류(goods_classify)가 의류인 데이터만 선택해보도록 하자.
select goods_name, goods_classify
from goods
where goods_classify = '의류';

## 연산자
### 산술 연산자
select *, sell_price - buy_price as profit
from goods
where sell_price - buy_price >= 500;

### 비교 연산자
-- sell_price가 1000 이상인 데이터만 선택하는 쿼리는 다음과 같다.
select goods_name, goods_classify, sell_price
from goods
where sell_price >= 1000;

-- 숫자 뿐 아니라 날짜에도 비교 연산자를 사용할 수 있다. 
-- 등록일(register_date)이 2020년 9월 27일 이전인 데이터만 선택하는 쿼리는 다음과 같다.
select goods_name, goods_classify, register_date
from goods
where register_date < '2020-09-27';

### 논리 연산자

-- where 구 내에 `and` 연산자와 `or` 연산자와 같은 논리 연산자를 사용하면 복수의 검색 조건을 조합할 수 있다. 
-- 예를 들어 상품 분류가 주방용품이고 판매가가 3000 이상인 데이터를 조회하는 쿼리는 다음과 같다.
select goods_name, goods_classify, sell_price
from goods
where goods_classify = '주방용품'
and sell_price >= 3000;

-- 상품 분류가 주방용품이거나 판매가가 3000 이상인 경우처럼 여러 조건 중 하나만 만족해도 되는 경우를 검색하고 싶을 경우에는 `or` 연산자를 사용하면 된다.
select goods_name, goods_classify, sell_price
from goods
where goods_classify = '주방용품'
or sell_price >= 3000;

## 집약 함수
### `count`: 행 숫자를 계산
-- `count` 함수는 행의 숫자를 계산한다. goods 테이블에 몇 개의 행이 있는 확인하는 쿼리는 다음과 같다.
select count(*)
from goods;

-- null을 제외한 행의 수를 계산하고자 할 때는 인수에 특정 열을 지정한다.
select count(buy_price)
from goods;

### `sum`: 합계를 계산
-- `sum` 함수는 특정 열의 합계를 계산하며, null 값은 무시하고 계산이 된다. sell_price와 buy_price 열의 합계를 구하는 쿼리는 다음과 같다.
select sum(sell_price), sum(buy_price)
from goods;

### `avg`: 산술평균을 계산
-- `avg` 함수는 산술평균을 구하며, 사용법은 `sum`과 동일하다. sell_price 열의 평균을 구하는 쿼리는 다음과 같다.
select avg(sell_price)
from goods;

### 중복값 제외 후 집약함수 사용하기
-- 만일 상품 분류가 몇 개가 있는지 확인하고 싶을 때는 어떻게 하면 될까? 
-- `count` 함수의 인자에 `distict` 키워드를 사용해 중복되지 않은 데이터의 갯수를 계산할 수 있다.
select count(distinct goods_classify) 
from goods;

## 그룹화와 정렬
### 그룹 나누기
-- 상품 분류 별 데이터의 수를 계산하기 위한 쿼리는 다음과 같다.
select goods_classify, count(*)
from goods
group by goods_classify;

-- `group by` 구는 반드시 `from` 구 뒤에 두어야 한다. 
-- 이번에는 buy_price 별 행 갯수를 구해보도록 하자.
select buy_price, count(*)
from goods
group by buy_price;

-- 만일 `where` 구를 통해 조건에 맞는 데이터를 선택한 후 `group by` 구를 통해 그룹을 나눌때는 어떻게 해야 할까? 
-- 이 경우 `where` 구 뒤에 `group by` 구를 작성해야 한다. 
-- 상품 분류가 의류인 것 중 buy_price 별 데이터의 수를 구하는 쿼리는 다음과 같다.
select buy_price, count(*)
from goods
where goods_classify = '의류'
group by buy_price;

-- `group by`를 통해 나온 결과에 조건을 지정하려면 어떻게 해야 할까? 
-- 이 경우 `where`이 아닌 `having` 구를 사용해야 한다.
-- 예를 들어 상품 분류별로 판매가의 평균을 구한 후, 이 값이 2500 이상인 데이터를 구하는 쿼리는 다음과 같다.
select goods_classify, avg(sell_price)
from goods
group by goods_classify
having avg(sell_price) >= 2500;


### 검색 결과 정렬하기
-- SQL에서는 결과가 무작위로 정렬되므로 쿼리를 실행할 때 마다 결과가 변한다. 
-- 오름차순이나 내림차순으로 결과를 정렬하고자 할 경우에는 `order by` 구를 사용한다.
-- 예를 들어 sell_price가 싼 순서, 즉 오름차순으로 정렬할 경우 쿼리는 다음과 같다.
select *
from goods
order by sell_price;

-- 내림차순으로 정렬하고자 할 경우 재정렬 기준 뒤에 desc 키워드를 사용한다.
select *
from goods
order by sell_price desc;

## 뷰와 서브쿼리
### 뷰 만들기
-- 뷰는 데이터를 저장하지 않고 있으며, 뷰에서 데이터를 꺼내려고 할 때 내부적으로 쿼리를 실행하여 일시적인 가상 테이블을 만든다. 
-- 즉, 데이터가 아닌 쿼리를 저장하고 있다고 보면 된다. 이러한 뷰가 가진 장점은 다음과 같다.
-- 1. 데이터를 저장하지 않기 때문에 기억 장치 용량을 절약할 수 있다.
-- 2. 자주 사용하는 쿼리를 매번 작성하지 않고 뷰로 저장하면 반복해서 사용이 가능한다. 
-- 뷰는 원래의 테이블과 연동되므로, 데이터가 최신 상태로 갱신되면 뷰의 결과 역시 자동으로 최신 상태를 보여준다.
-- 만일 **상품 분류 별 행 갯수**를 매일 조회해야 한다면, 매번 쿼리를 실행하는 것 보다 뷰를 만들어 이를 확인하는 것이 훨씬 효율적이다. 
-- 아래의 쿼리를 통해 해당 뷰를 만들 수 있다.
create view GoodSum (goods_classify, cnt_goods)
as
select goods_classify, count(*)
from goods
group by goods_classify;

-- 뷰의 데이터를 확인하는 방법은 테이블의 데이터를 확인하는 방법과 동일하다.
select *
from GoodSum;

### 뷰 삭제하기
drop view GoodSum;

### 서브쿼리
-- 서브쿼리란 쿼리 내의 쿼리이며, 일회용 뷰를 의미한다. 
-- 즉, 뷰를 정의하는 구문을 그대로 다른 구 안에 삽입하는 것이다. 
-- 먼저 뷰를 만든 후 이를 확인하는 쿼리는 다음과 같다.

select goods_classify, cnt_goods
from (
 select goods_classify, count(*) as cnt_goods
 from goods
 group by goods_classify
) as GoodsSum;

### 스칼라 서브쿼리
-- 스칼라 서브쿼리란 단이 값이 반환되는 서브쿼리다. 
-- 판매단가가 전체 평균 판매단가보다 높은 상품만을 검색
select avg(sell_price)
from goods;

select *
from goods
where sell_price > (select avg(sell_price) from goods);

-- 평균 판매가격을 새로운 열로 만드는 쿼리는 다음과 같다.
select goods_id, goods_name, sell_price,
	(select avg(sell_price) from goods) as avg_price
from goods;

-- 상품 분류 별 평균 판매가격이 전체 데이터의 평균 판매가격 이상인 데이터만 출력하는 쿼리
select goods_classify, avg(sell_price)
from goods
group by goods_classify
having avg(sell_price) > (select avg(sell_price) from goods);

## 함수, 술어와 case 식
### 산술 함수

create table SampleMath
(m  numeric (10,3),
 n  integer,
 p  integer);
 
insert into SampleMath(m, n, p) values (500, 0, NULL);
insert into SampleMath(m, n, p) values (-180, 0, NULL);
insert into SampleMath(m, n, p) values (NULL, NULL, NULL);
insert into SampleMath(m, n, p) values (NULL, 7, 3);
insert into SampleMath(m, n, p) values (NULL, 5, 2);
insert into SampleMath(m, n, p) values (NULL, 4, NULL);
insert into SampleMath(m, n, p) values (8, NULL, 3);
insert into SampleMath(m, n, p) values (2.27, 1, NULL);
insert into SampleMath(m, n, p) values (5.555,2, NULL);
insert into SampleMath(m, n, p) values (NULL, 1, NULL);
insert into SampleMath(m, n, p) values (8.76, NULL, NULL);

select * from SampleMath;

#### `abs`: 절대값 계산하기
select m, abs(m) as abs_m
from SampleMath;

#### `mod`: 나눗셈의 나머지 구하기
-- 7 나누기 3의 몫은 2이며 나머지는 1이다. `mod` 함수는 이 나머지에 해당하는 값을 구해준다.
select n, p, mod(n, p) as mod_col
from SampleMath;

### `round`: 반올림 하기
-- `round` 함수를 통해 반올림을 할 수 있으며, 몇 째자리에서 반올림을 할지 정할 수 있다.
-- `round(m, 2)`의 경우 할 경우 m열의 데이터를 소수 둘째자리까지 반올림한다.
select m, n, round(m, n) as round_col
from SampleMath;

### 문자열 함수
create table SampleStr
(str1  varchar(40),
 str2  varchar(40),
 str3  varchar(40));
 
insert into SampleStr (str1, str2, str3) values ('가나다', '라마', NULL);
insert into SampleStr (str1, str2, str3) values ('abc', 'def', NULL);
insert into SampleStr (str1, str2, str3) values ('김', '철수', '입니다');
insert into SampleStr (str1, str2, str3) values ('aaa', NULL, NULL);
insert into SampleStr (str1, str2, str3) values (NULL, '가가가', NULL);
insert into SampleStr (str1, str2, str3) values ('@!#$%', NULL,	NULL);
insert into SampleStr (str1, str2, str3) values ('ABC',	NULL, NULL);
insert into SampleStr (str1, str2, str3) values ('aBC',	NULL, NULL);
insert into SampleStr (str1, str2, str3) values ('abc철수', 'abc', 'ABC');
insert into SampleStr (str1, str2, str3) values ('abcdefabc','abc', 'ABC');
insert into SampleStr (str1, str2, str3) values ('아이우', '이','우');

select * from SampleStr;

#### `concat`: 문자열 연결
-- `concat` 함수는 여러 열의 문자열을 연결하는데 사용됩니다. 
-- (타 RDMS에서는 ||로 문자를 합치기도 한다.) 먼저 str1과 str2 열의 문자를 합쳐보도록 하자.
select str1, str2, concat(str1, str2) as str_concat
from SampleStr;

#### `lower`: 소문자로 변환
select str1, lower(str1) as low_str
from SampleStr;

#### `replace`: 문자를 변경
-- `replace(대상 문자열, 치환 전 문자열, 치환 후 문자열)` 
select str1, str2, str3,
	replace(str1, str2, str3) as rep_str
from SampleStr;

### 날짜 함수
#### 현재 날짜, 시간, 일시
-- 현재 날짜(`current_date`)와 시간(`current_time`), 일시(`current_timestamp`)를 다루는 함수의 경우 from 구문이 없이 사용이 가능하다.
select current_date, current_time, current_timestamp;

#### 날짜 요소 추출하기
-- `extract(날짜 요소 from 날짜)` 함수를 통해 년, 월, 시, 초 등을 추출할 수 있다.
select
    current_timestamp,
    extract(year from current_timestamp) as year,
    extract(month from current_timestamp) as month,
    extract(day	from current_timestamp) as day,
    extract(hour from current_timestamp) as hour,
    extract(minute from current_timestamp) as minute,
    extract(second from current_timestamp) as second;
    
### 술어
-- 술어란 반환 값이 진리값(TRUE, FALSE, UNKNOWN)인 함수를 가리킨다. 
-- 대표적인 예로는 `like`, `between`, `is null`, `in` 등이 있다.
create table SampleLike
(strcol varchar(6) not null,
primary key (strcol));

insert into SampleLike (strcol) values ('abcddd');
insert into SampleLike (strcol) values ('dddabc');
insert into SampleLike (strcol) values ('abdddc');
insert into SampleLike (strcol) values ('abcdd');
insert into SampleLike (strcol) values ('ddabc');
insert into SampleLike (strcol) values ('abddc');

select * from SampleLike;   

#### `like`: 문자열 부분 일치
-- 앞에서 문자열을 검색할 때는 등호(=)를 사용했지만, 이는 완전히 일치하는 경우에만 참이 된다. 
-- 반면 `like` 술어는 문자열 중 **부분 일치를 검색**할 때 사용한다. 먼저 아래의 테이블을 만들도록 한다.
-- `%`는 '0문자 이상의 임의 문자열' 을 의미하는 특수 기호이며, 아래의 예에서 'ddd%'는 'ddd로 시작하는 모든 문자열'을 의미한다. 
select *
from SampleLike
where strcol like 'ddd%';

-- '%ddd%' 처럼 문자열 처음과 끝을 %로 감쌀 경우 '문자열 안에 ddd를 포함하고 있는 모든 문자열'
select *
from SampleLike
where strcol like '%ddd%';

-- 후방 일치 검색
select *
from SampleLike
where strcol like '%ddd';

#### `between`: 범위 검색
-- `between`은 범위 검색을 수행한다. 
-- goods 테이블에서 sell_price가 100원부터 1000원까지인 상품을 선택할 때 `between` 술어를 사용하면 다음과 같이 나타낼 수 있다.
select *
from goods
where sell_price between 100 and 1000;

#### `is null`, `is not null`: null 데이터 선택
-- `where buy_price == null` 형식으로 작성하면 될 듯 하지만 해당 쿼리를 실행하면 오류가 발생
select *
from goods
where buy_price is null;

-- null이 포함되지 않은 데이터만 선택
select *
from goods
where buy_price is not null;

#### `in`: 복수의 값을 지정
-- 만일 buy_price가 320, 500, 5000인 상품을 선택할 경우, or을 쓰면 다음과 같이 쿼리를 작성해야 한다.
select *
from goods
where buy_price = 320 
	or buy_price = 500
	or buy_price = 5000;

-- 이러한 나열식의 쿼리는 조건이 많아질수록 길어지고 효율성이 떨어진다. 
-- 이 때 사용할 수 있는 것이 `in` 술어로써 `in(값 1, 값 2, …)` 형태를 통해 간단하게 표현할 수 있다.	
select *
from goods
where buy_price in (320, 500, 5000);

-- 반대로 buy_price가 320, 500, 5000이 아닌 데이터만 선택하고 싶을 때는 `not in` 술어를 사용한다.
select *
from goods
where buy_price not in (320, 500, 5000);

### `case` 식
-- sell_price가 6000 이상이면 고가, 3000과 6000 사이면 저가, 3000 미만이면 저가로 구분할 경우, `case`식을 사용하여 각 평가식에 따른 구분을 할 수 있다. 
select goods_name, sell_price,
	case when sell_price >=  6000 then '고가'    
		 when sell_price >= 3000 and sell_price < 6000 then '중가'
         when sell_price < 3000 then '저가'
		 else null
end as price_classify
from goods;

## 테이블의 집합과 결합

CREATE TABLE goods2
(goods_id CHAR(4) NOT NULL,
 goods_name VARCHAR(100) NOT NULL,
 goods_classify VARCHAR(32) NOT NULL,
 sell_price INTEGER,
 buy_price INTEGER,
 register_date DATE,
 PRIMARY KEY (goods_id));

insert into goods2 values ('0001', '티셔츠' ,'의류', 1000, 500, '2020-09-20');
insert into goods2 values ('0002', '펀칭기', '사무용품', 500, 320, '2020-09-11');
insert into goods2 values ('0003', '와이셔츠', '의류', 4000, 2800, NULL);
insert into goods2 values ('0009', '장갑', '의류', 800, 500, NULL);
insert into goods2 values ('0010', '주전자', '주방용품', 2000, 1700, '2020-09-20');

select * from goods2;
### 테이블 더하기
-- 기존의 Goods 테이블과 새로 만든 Goods2 테이블을 위아래로 합쳐보도록 한다. 
select *
from goods
union
select *
from goods2;

-- 상품ID가 001, 002, 003인 3개 행은 양쪽 테이블에 모두 존재하며, `union` 구문에서는 중복행을 제외하고 테이블을 합쳤다. 
-- 만일 중복 행을 포함하여 테이블을 합치고자 할 경우에는 `union all` 구문을 사용하면 된다.
select *
from goods
union all
select *
from goods2;

### 테이블 결합
CREATE TABLE StoreGoods
(store_id CHAR(4) NOT NULL,
 store_name VARCHAR(200) NOT NULL,
 goods_id CHAR(4) NOT NULL,
 num INTEGER NOT NULL,
 PRIMARY KEY (store_id, goods_id));

insert into StoreGoods (store_id, store_name, goods_id, num) values ('000A', '서울',	'0001',	30);
insert into StoreGoods (store_id, store_name, goods_id, num) values ('000A', '서울',	'0002',	50);
insert into StoreGoods (store_id, store_name, goods_id, num) values ('000A', '서울',	'0003',	15);
insert into StoreGoods (store_id, store_name, goods_id, num) values ('000B', '대전',	'0002',	30);
insert into StoreGoods (store_id, store_name, goods_id, num) values ('000B',' 대전',	'0003',	120);
insert into StoreGoods (store_id, store_name, goods_id, num) values ('000B', '대전',	'0004',	20);
insert into StoreGoods (store_id, store_name, goods_id, num) values ('000B', '대전',	'0006',	10);
insert into StoreGoods (store_id, store_name, goods_id, num) values ('000B', '대전',	'0007',	40);
insert into StoreGoods (store_id, store_name, goods_id, num) values ('000C', '부산',	'0003',	20);
insert into StoreGoods (store_id, store_name, goods_id, num) values ('000C', '부산',	'0004',	50);
insert into StoreGoods (store_id, store_name, goods_id, num) values ('000C', '부산',	'0006',	90);
insert into StoreGoods (store_id, store_name, goods_id, num) values ('000C', '부산',	'0007',	70);
insert into StoreGoods (store_id, store_name, goods_id, num) values ('000D', '대구',	'0001',	100);

#### `inner join`: 내부 결합
-- goods_id는 두 테이블에 모두 존재하며, 다른 열들은 한쪽 테이블에만 존재한다. 
-- goods_id를 기준으로 StoreGoods 테이블에 Goods 테이블을 결합하는 방법
select store.store_id, store.store_name, store.goods_id,
	goods.goods_name, goods.sell_price
from StoreGoods as store 
inner join goods as goods
	on store.goods_id = goods.goods_id;

#### `outer join` 외부 결합
-- inner join은 두 테이블에 모두 존재하는 데이터를 합쳤지만, outer join은 한쪽 테이블에만 존재하는 데이터도 출력한다.
-- 먼저 StoreGoods와 Goods에 존재하는 상품ID를 검색한다.
select distinct(goods_id) from StoreGoods;
select distinct(goods_id) from goods;

-- StoreGoods 1~4, 6~7번이, Goods 1번부터 8번까지 상품이 있다. 
-- 즉, StoreGoods 5번(압력솥)과 8번(볼펜) ID에 해당하는 물건이 없다. 이제 `outer join`을 해보도록 한다.
-- `outer join`은 한쪽 테이블에만 존재해도 누락 없이 모두 출력하며, 정보가 없는 부분은 NULL로 표시
select store.store_id, store.store_name, goods.goods_id,
	goods.goods_name, goods.sell_price
from StoreGoods as store 
right outer join goods as goods
	on store.goods_id = goods.goods_id;

-- goods_id를 기준으로 왼쪽에 해당하는 StoreGoods 테이블의 내용이 결합되었다. 위 쿼리를 `left outer join` 으로 바꿔보도록 하자.
select store.store_id, store.store_name, goods.goods_id,
	goods.goods_name, goods.sell_price
from StoreGoods as store 
left outer join goods as goods
	on store.goods_id = goods.goods_id;
	
## SQL 고급 처리
### 윈도우 함수
-- 윈도우 함수를 이용하면 랭킹, 순번 생성 등 일반적인 집약 함수로는 불가능한 고급처리를 할 수 있다.
#### `rank`: 순위를 계산
-- 예를 들어 Goods 테이블의 상품 중 상품분류(goods_classify) 별로 판매단가(sell_price)가 낮은 순서대로 순위를 구하는 방법은 다음과 같다.
-- 1. `partition by`는 순위를 정할 대상 범위를 설정하며, 어떤 조건으로 그룹을 나눈다고 생각하면 이해가 쉽다. 상품 분류마다 순위를 구하고자 하므로 goods_classify를 입력한다.
-- 2. `order by`는 윈도우 함수를 어떤 열에 어떤 순서로 적용할지 정한다. 판매단가를 오름차순으로 순위를 구하고자 하므로 sell_price를 입력하였다. 만일 내림차순으로 순위를 구하고자 할 경우 `desc` 를 입력하면 된다. (기본적으로 asc 즉 오름차순이 적용된다.)
-- 3. 순위를 구하는 윈도우 전용 함수인 `rank()`를 입력한다. 
select goods_name, goods_classify, sell_price,
	rank() over (partition by goods_classify order by sell_price) as ranking
from goods;

-- `partition by`를 지정하지 않으면 전체 테이블이 윈도우가 되므로, 아래와 같이 sell_price 열 자체를 기준으로 순위가 구해진다.
select goods_name, goods_classify, sell_price,
	rank () over (order by sell_price) as ranking
from goods; 

-- `rank`: 같은 순위인 행이 복수개 있으면 후순위를 건너뛴다. 예) 1위가 3개인 경우: 1위, 1위, 1위, 4위, …
-- `dense_rank`: 같은 순위인 행이 복수가 있어도 후순위를 건너뛰지 않는다. 예) 1위가 3개인 경우: 1위, 1위, 1위, 2위, …
-- `row_number`: 순위와 상관없이 연속 번호를 부여한다. 예: 1위가 3개인 레코드인 경우: 1위, 2위, 3위, 4위, …
select goods_name, goods_classify, sell_price,
	rank() over (order by sell_price) as ranking,
    dense_rank() over (order by sell_price) as ranking,
    row_number() over (order by sell_price) as ranking
from goods;

#### 윈도우 함수에서 집약 함수의 사용
-- `over()`를 빈 칸으로 둘 경우 current_sum 열에는 모든 sell_price의 합계가 나타난다.
select goods_id, goods_name, sell_price,
	sum(sell_price) over() as current_sum
from goods;

-- 이번에는 goods_id에 따른 누적합계를 구해보도록 하자.
select goods_id, goods_name, sell_price,
	sum(sell_price) over(order by goods_id) as current_sum
from goods;

-- goods_id에 따른 누적평균를 구해보도록 하자.
select goods_id, goods_name, sell_price,
	avg(sell_price) over(order by goods_id) as current_avg
from goods;

-- `partition by`를 추가하면 윈도우 별로 집계도 가능하다.
select goods_id, goods_classify, goods_name, sell_price,
	sum(sell_price) over(partition by goods_classify order by goods_id) as current_sum
from goods;

#### 이동평균 계산하기
-- 윈도우 함수에서는 그 범위를 정해 '프레임'을 만들 수도 있다. 
-- 이는 `over` 내의 `order by` 구문 뒤에 범위 지정 키워드를 사용하면 된다. 
-- 예를 들어 모든 열에 대한 누적평균이 아닌 최근 3개 데이터만 이용해 평균을 구하는 이동평균을 계산하는 쿼리는 다음과 같다.
-- `rows n proceding`을 입력할 경우 앞의 n 행까지만 프레임을 만들어 계산한다. 위 예제에서는 n=2를 입력했으므로 현재 행과 앞의 두개 행, 즉 3개 행으로만 이동평균을 계산한다.
select goods_id, goods_classify, goods_name, sell_price,
	avg(sell_price) over(order by goods_id rows 2 preceding) as moving_avg
from goods;

-- 앞의 행이 아닌 뒤의 행을 이용해 계산하고 싶을 경우 `preceding` 대신 `following`을 입력한다. 현재 행과 뒤의 두개 행으로 이동평균을 계산하는 법은 다음과 같다.
-- `current row and 2 following`는 현재 행과 뒤의 두개 행을 의미하며, 앞서 살펴 본 `preceding`과 반대로 뒤에서 부터 이동평균의 계산된다. 
select goods_id, goods_classify, goods_name, sell_price,
	avg(sell_price) over(order by goods_id rows between current row and 2 following) as moving_avg
from goods;

-- `preceding`과 `following`을 동시에 사용하는 것도 가능하다.
-- `rows between n preceding and m following`을 입력하면 앞의 n행과 뒤의 m행 까지를 프레임으로 지정한다. 위의 예에서는 앞의 1개 행과 뒤의 1개 행, 총 3개 행을 이용해 이동평균이 계산된다
select goods_id, goods_classify, goods_name, sell_price,
	avg(sell_price) over(order by goods_id
    rows between 1 preceding and 1 following)
    as moving_avg
from goods;
