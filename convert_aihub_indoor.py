import os
import json
import pickle
from datetime import datetime
import glob
import shutil
from tqdm import tqdm
import csv
import numpy as np

# 카테고리 정의 (MANUFACT_CATEGORIES 또는 INDOOR_CATEGORIES 사용)
MANUFACT_CATEGORIES = [
    {"supercategory": "도구 및 장비", "id": 1, "name": "가스디퓨저"},
    {"supercategory": "도구 및 장비", "id": 2, "name": "가스토치"},
    {"supercategory": "도구 및 장비", "id": 3, "name": "간극 게이지"},
    {"supercategory": "도구 및 장비", "id": 4, "name": "구리스건"},
    {"supercategory": "도구 및 장비", "id": 5, "name": "그랩훅"},
    {"supercategory": "도구 및 장비", "id": 6, "name": "기어풀러"},
    {"supercategory": "도구 및 장비", "id": 7, "name": "스크레퍼(끌->스크레퍼로 명칭 변경)"},
    {"supercategory": "도구 및 장비", "id": 8, "name": "납흡입기"},
    {"supercategory": "도구 및 장비", "id": 9, "name": "니퍼"},
    {"supercategory": "도구 및 장비", "id": 10, "name": "대패"},
    {"supercategory": "도구 및 장비", "id": 11, "name": "덕트테이프"},
    {"supercategory": "도구 및 장비", "id": 12, "name": "도르레"},
    {"supercategory": "도구 및 장비", "id": 13, "name": "도배칼"},
    {"supercategory": "도구 및 장비", "id": 14, "name": "드라이버"},
    {"supercategory": "도구 및 장비", "id": 15, "name": "드릴 지그"},
    {"supercategory": "도구 및 장비", "id": 16, "name": "레이저 거리 측정기"},
    {"supercategory": "도구 및 장비", "id": 17, "name": "리베터기"},
    {"supercategory": "도구 및 장비", "id": 18, "name": "만력기"},
    {"supercategory": "도구 및 장비", "id": 19, "name": "망치"},
    {"supercategory": "도구 및 장비", "id": 20, "name": "멀티미터"},
    {"supercategory": "도구 및 장비", "id": 21, "name": "몽키 스패너"},
    {"supercategory": "도구 및 장비", "id": 22, "name": "미장칼"},
    {"supercategory": "도구 및 장비", "id": 23, "name": "바이스그립"},
    {"supercategory": "도구 및 장비", "id": 24, "name": "밴드쏘"},
    {"supercategory": "도구 및 장비", "id": 25, "name": "버니어 캘리퍼스"},
    {"supercategory": "도구 및 장비", "id": 26, "name": "볼트 커터"},
    {"supercategory": "도구 및 장비", "id": 27, "name": "분도기"},
    {"supercategory": "도구 및 장비", "id": 28, "name": "분사기"},
    {"supercategory": "도구 및 장비", "id": 29, "name": "브러쉬"},
    {"supercategory": "도구 및 장비", "id": 30, "name": "삼각자"},
    {"supercategory": "도구 및 장비", "id": 31, "name": "삽"},
    {"supercategory": "도구 및 장비", "id": 32, "name": "소음계"},
    {"supercategory": "도구 및 장비", "id": 33, "name": "수평계"},
    {"supercategory": "도구 및 장비", "id": 34, "name": "스크류잭"},
    {"supercategory": "도구 및 장비", "id": 35, "name": "스트립퍼"},
    {"supercategory": "도구 및 장비", "id": 36, "name": "스패너"},
    {"supercategory": "도구 및 장비", "id": 37, "name": "슬링훅"},
    {"supercategory": "도구 및 장비", "id": 38, "name": "시멘트교반기"},
    {"supercategory": "도구 및 장비", "id": 39, "name": "앵글 그라인더"},
    {"supercategory": "도구 및 장비", "id": 40, "name": "양구스패너"},
    {"supercategory": "도구 및 장비", "id": 41, "name": "연무기"},
    {"supercategory": "도구 및 장비", "id": 42, "name": "열풍기"},
    {"supercategory": "도구 및 장비", "id": 43, "name": "오링풀러"},
    {"supercategory": "도구 및 장비", "id": 44, "name": "용접홀더"},
    {"supercategory": "도구 및 장비", "id": 45, "name": "유리칼"},
    {"supercategory": "도구 및 장비", "id": 46, "name": "유리흡착기"},
    {"supercategory": "도구 및 장비", "id": 47, "name": "유압절단기"},
    {"supercategory": "도구 및 장비", "id": 48, "name": "육각 소켓 렌치"},
    {"supercategory": "도구 및 장비", "id": 49, "name": "육각렌치"},
    {"supercategory": "도구 및 장비", "id": 50, "name": "인두기"},
    {"supercategory": "도구 및 장비", "id": 51, "name": "임팩트랜치"},
    {"supercategory": "도구 및 장비", "id": 52, "name": "적외선온도계"},
    {"supercategory": "도구 및 장비", "id": 53, "name": "전동드릴"},
    {"supercategory": "도구 및 장비", "id": 54, "name": "절곡집게"},
    {"supercategory": "도구 및 장비", "id": 55, "name": "접이톱"},
    {"supercategory": "도구 및 장비", "id": 56, "name": "접지봉커넥터"},
    {"supercategory": "도구 및 장비", "id": 57, "name": "줄톱"},
    {"supercategory": "도구 및 장비", "id": 58, "name": "직쏘"},
    {"supercategory": "도구 및 장비", "id": 59, "name": "체인톱"},
    {"supercategory": "도구 및 장비", "id": 60, "name": "콤비네이션스퀘어"},
    {"supercategory": "도구 및 장비", "id": 61, "name": "타일절단기"},
    {"supercategory": "도구 및 장비", "id": 62, "name": "타카"},
    {"supercategory": "도구 및 장비", "id": 63, "name": "테이퍼게이지"},
    {"supercategory": "도구 및 장비", "id": 64, "name": "토크렌치"},
    {"supercategory": "도구 및 장비", "id": 65, "name": "톱"},
    {"supercategory": "도구 및 장비", "id": 66, "name": "파이프랜치"},
    {"supercategory": "도구 및 장비", "id": 67, "name": "파이프밴더"},
    {"supercategory": "도구 및 장비", "id": 68, "name": "파이프커터"},
    {"supercategory": "도구 및 장비", "id": 69, "name": "파이프확관기"},
    {"supercategory": "도구 및 장비", "id": 70, "name": "팜맥"},
    {"supercategory": "도구 및 장비", "id": 71, "name": "펜치"},
    {"supercategory": "도구 및 장비", "id": 72, "name": "프라이바"},
    {"supercategory": "도구 및 장비", "id": 73, "name": "플라이어"},
    {"supercategory": "도구 및 장비", "id": 74, "name": "플라이어 첼라"},
     {"supercategory": "도구 및 장비", "id": 75, "name": "핀게이지"},
    {"supercategory": "도구 및 장비", "id": 76, "name": "하이트게이지"},
    {"supercategory": "도구 및 장비", "id": 77, "name": "함마렌치"},
    {"supercategory": "도구 및 장비", "id": 78, "name": "항공가위"},
    {"supercategory": "도구 및 장비", "id": 79, "name": "도끼"},
    {"supercategory": "도구 및 장비", "id": 80, "name": "훅스패너"},
    {"supercategory": "도구 및 장비", "id": 81, "name": "힌지핸들"},
    {"supercategory": "도구 및 장비", "id": 82, "name": "T핸들"},
    {"supercategory": "자재 및 부품", "id": 83, "name": "45도 엘보 파이프"},
    {"supercategory": "자재 및 부품", "id": 84, "name": "90도 엘보 파이프"},
    {"supercategory": "자재 및 부품", "id": 85, "name": "각재"},
    {"supercategory": "자재 및 부품", "id": 86, "name": "경첩"},
    {"supercategory": "자재 및 부품", "id": 87, "name": "고압호스"},
    {"supercategory": "자재 및 부품", "id": 88, "name": "8자브라켓"},
    {"supercategory": "자재 및 부품", "id": 89, "name": "다목적가위"},
    {"supercategory": "자재 및 부품", "id": 90, "name": "도어 스토퍼"},
    {"supercategory": "자재 및 부품", "id": 91, "name": "도장용마스킹테이프"},
    {"supercategory": "자재 및 부품", "id": 92, "name": "라벨용지"},
    {"supercategory": "자재 및 부품", "id": 93, "name": "롤러 베어링"},
    {"supercategory": "자재 및 부품", "id": 94, "name": "금형스프링"},
    {"supercategory": "자재 및 부품", "id": 95, "name": "매직케이블"},
    {"supercategory": "자재 및 부품", "id": 96, "name": "바 클램프"},
    {"supercategory": "자재 및 부품", "id": 97, "name": "T조인트"},
    {"supercategory": "자재 및 부품", "id": 98, "name": "방청제"},
    {"supercategory": "자재 및 부품", "id": 99, "name": "밸브"},
    {"supercategory": "자재 및 부품", "id": 100, "name": "베벨기어"},
    {"supercategory": "자재 및 부품", "id": 101, "name": "베어링 플레이트"},
    {"supercategory": "자재 및 부품", "id": 102, "name": "캐스터바퀴"},
    {"supercategory": "자재 및 부품", "id": 103, "name": "보드마카"},
    {"supercategory": "자재 및 부품", "id": 104, "name": "볼베어링"},
    {"supercategory": "자재 및 부품", "id": 105, "name": "볼스크류"},
    {"supercategory": "자재 및 부품", "id": 106, "name": "볼트"},
    {"supercategory": "자재 및 부품", "id": 107, "name": "브라켓"},
    {"supercategory": "자재 및 부품", "id": 108, "name": "샤프트"},
    {"supercategory": "자재 및 부품", "id": 109, "name": "샤프트 홀더"},
    {"supercategory": "자재 및 부품", "id": 110, "name": "유량계"},
    {"supercategory": "자재 및 부품", "id": 111, "name": "스파이러 스프링"},
    {"supercategory": "자재 및 부품", "id": 112, "name": "슬라이드 레일"},
    {"supercategory": "자재 및 부품", "id": 113, "name": "실리콘실란트"},
    {"supercategory": "자재 및 부품", "id": 114, "name": "십자형 접속 파이프"},
    {"supercategory": "자재 및 부품", "id": 115, "name": "액체풀"},
    {"supercategory": "자재 및 부품", "id": 116, "name": "연마석"},
    {"supercategory": "자재 및 부품", "id": 117, "name": "용접자석"},
    {"supercategory": "자재 및 부품", "id": 118, "name": "컴프레셔 안전핀"},
    {"supercategory": "자재 및 부품", "id": 119, "name": "웜기어"},
    {"supercategory": "자재 및 부품", "id": 120, "name": "전구"},
    {"supercategory": "자재 및 부품", "id": 121, "name": "전선관"},
    {"supercategory": "자재 및 부품", "id": 122, "name": "절연테이프"},
    {"supercategory": "자재 및 부품", "id": 123, "name": "접착제"},
    {"supercategory": "자재 및 부품", "id": 124, "name": "에어레귤레이터"},
    {"supercategory": "자재 및 부품", "id": 125, "name": "축전기"},
    {"supercategory": "자재 및 부품", "id": 126, "name": "컷쏘날"},
    {"supercategory": "자재 및 부품", "id": 127, "name": "케이블"},
    {"supercategory": "자재 및 부품", "id": 128, "name": "코너비드"},
    {"supercategory": "자재 및 부품", "id": 129, "name": "코일 스프링"},
    {"supercategory": "자재 및 부품", "id": 130, "name": "퀵클램프"},
    {"supercategory": "자재 및 부품", "id": 131, "name": "타일"},
    {"supercategory": "자재 및 부품", "id": 132, "name": "파스너"},
    {"supercategory": "자재 및 부품", "id": 133, "name": "펀치"},
    {"supercategory": "자재 및 부품", "id": 134, "name": "평기어"},
    {"supercategory": "자재 및 부품", "id": 135, "name": "포스트잇"},
    {"supercategory": "자재 및 부품", "id": 136, "name": "주사용 납"},
    {"supercategory": "자재 및 부품", "id": 137, "name": "형광등 휴즈"},
    {"supercategory": "자재 및 부품", "id": 138, "name": "헬리컬 기어(-> 배선 차단기)"},
    {"supercategory": "자재 및 부품", "id": 139, "name": "호스"},
    {"supercategory": "자재 및 부품", "id": 140, "name": "C형클램프"},
    {"supercategory": "자재 및 부품", "id": 141, "name": "dc모터"},
    {"supercategory": "자재 및 부품", "id": 142, "name": "L형클램프"},
    {"supercategory": "자재 및 부품", "id": 143, "name": "T형 접속 파이프"},
    {"supercategory": "자재 및 부품", "id": 144, "name": "usb-c 파워 어댑터"},
    {"supercategory": "자재 및 부품", "id": 145, "name": "Y형 접속 파이프"},
    {"supercategory": "보관 및 포장", "id": 146, "name": "가위"},
    {"supercategory": "보관 및 포장", "id": 147, "name": "글루건"},
    {"supercategory": "보관 및 포장", "id": 148, "name": "단프라 박스"},
    {"supercategory": "보관 및 포장", "id": 149, "name": "라벨프린터"},
    {"supercategory": "보관 및 포장", "id": 150, "name": "문서재단기"},
    {"supercategory": "보관 및 포장", "id": 151, "name": "박스 테이프"},
    {"supercategory": "보관 및 포장", "id": 152, "name": "노끈"},
    {"supercategory": "보관 및 포장", "id": 153, "name": "스티로폼 박스"},
    {"supercategory": "보관 및 포장", "id": 154, "name": "실리카겔"},
    {"supercategory": "보관 및 포장", "id": 155, "name": "실리콘건"},
    {"supercategory": "보관 및 포장", "id": 156, "name": "아이스팩"},
    {"supercategory": "보관 및 포장", "id": 157, "name": "에어캡"},
    {"supercategory": "보관 및 포장", "id": 158, "name": "연필꽂이"},
    {"supercategory": "보관 및 포장", "id": 159, "name": "인덱스카드"},
    {"supercategory": "보관 및 포장", "id": 160, "name": "제침기"},
    {"supercategory": "보관 및 포장", "id": 161, "name": "종이 박스"},
    {"supercategory": "보관 및 포장", "id": 162, "name": "종이 완충제"},
    {"supercategory": "보관 및 포장", "id": 163, "name": "줄자"},
    {"supercategory": "보관 및 포장", "id": 164, "name": "커터칼"},
    {"supercategory": "보관 및 포장", "id": 165, "name": "테이프커터"},
    {"supercategory": "보관 및 포장", "id": 166, "name": "파일"},
    {"supercategory": "보관 및 포장", "id": 167, "name": "핸드 홀 펀"},
    {"supercategory": "안전 및 보호", "id": 168, "name": "교통삼각대"},
    {"supercategory": "안전 및 보호", "id": 169, "name": "귀덮개"},
    {"supercategory": "안전 및 보호", "id": 170, "name": "귀마개"},
    {"supercategory": "안전 및 보호", "id": 171, "name": "나침반"},
    {"supercategory": "안전 및 보호", "id": 172, "name": "메가폰"},
    {"supercategory": "안전 및 보호", "id": 173, "name": "방독마스크"},
    {"supercategory": "안전 및 보호", "id": 174, "name": "방진마스크"},
    {"supercategory": "안전 및 보호", "id": 175, "name": "보안경"},
    {"supercategory": "안전 및 보호", "id": 176, "name": "보안면"},
    {"supercategory": "안전 및 보호", "id": 177, "name": "소화기"},
    {"supercategory": "안전 및 보호", "id": 178, "name": "라바콘"},
    {"supercategory": "안전 및 보호", "id": 179, "name": "스톱워치"},
    {"supercategory": "안전 및 보호", "id": 180, "name": "신호봉"},
    {"supercategory": "안전 및 보호", "id": 181, "name": "안전모"},
    {"supercategory": "안전 및 보호", "id": 182, "name": "안전화"},
    {"supercategory": "안전 및 보호", "id": 183, "name": "작업등"},
    {"supercategory": "안전 및 보호", "id": 184, "name": "작업용우의"},
    {"supercategory": "안전 및 보호", "id": 185, "name": "장갑"},
    {"supercategory": "안전 및 보호", "id": 186, "name": "헤드랜턴"},
    {"supercategory": "기타 물품", "id": 187, "name": "도구 및 장비함"},
    {"supercategory": "기타 물품", "id": 188, "name": "멀티탭"},
    {"supercategory": "기타 물품", "id": 189, "name": "무전기"},
    {"supercategory": "기타 물품", "id": 190, "name": "바코드스캐너"},
    {"supercategory": "기타 물품", "id": 191, "name": "쌍안경"},
    {"supercategory": "기타 물품", "id": 192, "name": "자물쇠"},
    {"supercategory": "기타 물품", "id": 193, "name": "저울"},
    {"supercategory": "기타 물품", "id": 194, "name": "전선거치대"},
    {"supercategory": "기타 물품", "id": 195, "name": "페인트롤러"},
    {"supercategory": "기타 물품", "id": 196, "name": "페인트붓"},
    {"supercategory": "기타 물품", "id": 197, "name": "플라스틱 바구니"},
    {"supercategory": "기타 물품", "id": 198, "name": "헤드셋"},
    {"supercategory": "기타 물품", "id": 199, "name": "호루라기"},
    {"supercategory": "기타 물품", "id": 200, "name": "확대경"}
]

INDOOR_CATEGORIES = [
    {"supercategory": "생활용품", "id": 1, "name": "갑티슈"},
    {"supercategory": "생활용품", "id": 2, "name": "건전지"},
    {"supercategory": "생활용품", "id": 3, "name": "기저귀"},
    {"supercategory": "생활용품", "id": 4, "name": "노트북"},
    {"supercategory": "생활용품", "id": 5, "name": "눈썹칼"},
    {"supercategory": "생활용품", "id": 6, "name": "다리미"},
    {"supercategory": "생활용품", "id": 7, "name": "달력"},
    {"supercategory": "생활용품", "id": 8, "name": "도끼빗"},
    {"supercategory": "생활용품", "id": 9, "name": "두루마리 휴지"},
    {"supercategory": "생활용품", "id": 10, "name": "드라이어"},
    {"supercategory": "생활용품", "id": 11, "name": "딱풀"},
    {"supercategory": "생활용품", "id": 12, "name": "로션"},
    {"supercategory": "생활용품", "id": 13, "name": "면도기"},
    {"supercategory": "생활용품", "id": 14, "name": "면봉"},
    {"supercategory": "생활용품", "id": 15, "name": "모니터"},
    {"supercategory": "생활용품", "id": 16, "name": "물티슈"},
    {"supercategory": "생활용품", "id": 17, "name": "바가지"},
    {"supercategory": "생활용품", "id": 18, "name": "바구니"},
    {"supercategory": "생활용품", "id": 19, "name": "바리깡"},
    {"supercategory": "생활용품", "id": 20, "name": "반창고"},
    {"supercategory": "생활용품", "id": 21, "name": "베개"},
    {"supercategory": "생활용품", "id": 22, "name": "병따개"},
    {"supercategory": "생활용품", "id": 23, "name": "보조배터리"},
    {"supercategory": "생활용품", "id": 24, "name": "분무기"},
    {"supercategory": "생활용품", "id": 25, "name": "브러쉬빗"},
    {"supercategory": "생활용품", "id": 26, "name": "블루투스 이어폰"},
    {"supercategory": "생활용품", "id": 27, "name": "빗자루"},
    {"supercategory": "생활용품", "id": 28, "name": "빨래집게"},
    {"supercategory": "생활용품", "id": 29, "name": "색연필"},
    {"supercategory": "생활용품", "id": 30, "name": "샤프"},
    {"supercategory": "생활용품", "id": 31, "name": "손톱깎이"},
    {"supercategory": "생활용품", "id": 32, "name": "스마트폰"},
    {"supercategory": "생활용품", "id": 33, "name": "스케치북"},
    {"supercategory": "생활용품", "id": 34, "name": "스테이플러"},
    {"supercategory": "생활용품", "id": 35, "name": "스프레이 살충제"},
    {"supercategory": "생활용품", "id": 36, "name": "스피커"},
    {"supercategory": "생활용품", "id": 37, "name": "습기제거제"},
    {"supercategory": "생활용품", "id": 38, "name": "시계"},
    {"supercategory": "생활용품", "id": 39, "name": "쓰레받기"},
    {"supercategory": "생활용품", "id": 40, "name": "아령"},
    {"supercategory": "생활용품", "id": 41, "name": "연필"},
    {"supercategory": "생활용품", "id": 42, "name": "연필깎이"},
    {"supercategory": "생활용품", "id": 43, "name": "염색빗"},
    {"supercategory": "생활용품", "id": 44, "name": "옷걸이"},
    {"supercategory": "생활용품", "id": 45, "name": "전화기"},
    {"supercategory": "생활용품", "id": 46, "name": "지우개"},
    {"supercategory": "생활용품", "id": 47, "name": "책"},
    {"supercategory": "생활용품", "id": 48, "name": "청소기"},
    {"supercategory": "생활용품", "id": 49, "name": "청소솔"},
    {"supercategory": "생활용품", "id": 50, "name": "충전기"},
    {"supercategory": "생활용품", "id": 51, "name": "커피포트"},
    {"supercategory": "생활용품", "id": 52, "name": "크레파스"},
    {"supercategory": "생활용품", "id": 53, "name": "키보드"},
    {"supercategory": "생활용품", "id": 54, "name": "태블릿PC"},
    {"supercategory": "생활용품", "id": 55, "name": "테이프"},
    {"supercategory": "생활용품", "id": 56, "name": "헤드폰"},
    {"supercategory": "생활용품", "id": 57, "name": "헤어구르프"},
    {"supercategory": "생활용품", "id": 58, "name": "형광펜"},
    {"supercategory": "생활용품", "id": 59, "name": "화분"},
    {"supercategory": "식품", "id": 60, "name": "계란"},
    {"supercategory": "식품", "id": 61, "name": "고구마"},
    {"supercategory": "식품", "id": 62, "name": "고추"},
    {"supercategory": "식품", "id": 63, "name": "당근"},
    {"supercategory": "식품", "id": 64, "name": "도넛"},
    {"supercategory": "식품", "id": 65, "name": "딸기"},
    {"supercategory": "식품", "id": 66, "name": "레몬"},
    {"supercategory": "식품", "id": 67, "name": "멜론"},
    {"supercategory": "식품", "id": 68, "name": "바나나"},
    {"supercategory": "식품", "id": 69, "name": "박스과자"},
    {"supercategory": "식품", "id": 70, "name": "버섯"},
    {"supercategory": "식품", "id": 71, "name": "버터"},
    {"supercategory": "식품", "id": 72, "name": "복숭아"},
    {"supercategory": "식품", "id": 73, "name": "봉지과자"},
    {"supercategory": "식품", "id": 74, "name": "브로콜리"},
    {"supercategory": "식품", "id": 75, "name": "사과"},
    {"supercategory": "식품", "id": 76, "name": "샌드위치"},
    {"supercategory": "식품", "id": 77, "name": "소시지"},
    {"supercategory": "식품", "id": 78, "name": "아보카도"},
    {"supercategory": "식품", "id": 79, "name": "양파"},
    {"supercategory": "식품", "id": 80, "name": "오렌지"},
    {"supercategory": "식품", "id": 81, "name": "요거트"},
    {"supercategory": "식품", "id": 82, "name": "우유"},
    {"supercategory": "식품", "id": 83, "name": "음료캔"},
    {"supercategory": "식품", "id": 84, "name": "참치캔"},
    {"supercategory": "식품", "id": 85, "name": "컵라면"},
    {"supercategory": "식품", "id": 86, "name": "케쳡"},
    {"supercategory": "식품", "id": 87, "name": "쿠키"},
    {"supercategory": "식품", "id": 88, "name": "크래커"},
    {"supercategory": "식품", "id": 89, "name": "크림치즈"},
    {"supercategory": "식품", "id": 90, "name": "통조림햄"},
    {"supercategory": "식품", "id": 91, "name": "파"},
    {"supercategory": "식품", "id": 92, "name": "팩주스"},
    {"supercategory": "식품", "id": 93, "name": "포도"},
    {"supercategory": "식품", "id": 94, "name": "햄버거"},
    {"supercategory": "식품", "id": 95, "name": "호박"},
    {"supercategory": "주방용품", "id": 96, "name": "감자칼"},
    {"supercategory": "주방용품", "id": 97, "name": "거품기"},
    {"supercategory": "주방용품", "id": 98, "name": "계량스푼"},
    {"supercategory": "주방용품", "id": 99, "name": "계량컵"},
    {"supercategory": "주방용품", "id": 100, "name": "국자"},
    {"supercategory": "주방용품", "id": 101, "name": "나이프"},
    {"supercategory": "주방용품", "id": 102, "name": "냄비"},
    {"supercategory": "주방용품", "id": 103, "name": "대접"},
    {"supercategory": "주방용품", "id": 104, "name": "도마"},
    {"supercategory": "주방용품", "id": 105, "name": "뒤집개"},
    {"supercategory": "주방용품", "id": 106, "name": "뚝배기"},
    {"supercategory": "주방용품", "id": 107, "name": "물통"},
    {"supercategory": "주방용품", "id": 108, "name": "병솔"},
    {"supercategory": "주방용품", "id": 109, "name": "빵칼"},
    {"supercategory": "주방용품", "id": 110, "name": "수세미"},
    {"supercategory": "주방용품", "id": 111, "name": "숟가락"},
    {"supercategory": "주방용품", "id": 112, "name": "스테인레스볼"},
    {"supercategory": "주방용품", "id": 113, "name": "식칼"},
    {"supercategory": "주방용품", "id": 114, "name": "식판"},
    {"supercategory": "주방용품", "id": 115, "name": "알루미늄 호일"},
    {"supercategory": "주방용품", "id": 116, "name": "어린이용 젓가락"},
    {"supercategory": "주방용품", "id": 117, "name": "얼음 트레이"},
    {"supercategory": "주방용품", "id": 118, "name": "와플기"},
    {"supercategory": "주방용품", "id": 119, "name": "쟁반"},
    {"supercategory": "주방용품", "id": 120, "name": "접시"},
    {"supercategory": "주방용품", "id": 121, "name": "젓가락"},
    {"supercategory": "주방용품", "id": 122, "name": "주걱"},
    {"supercategory": "주방용품", "id": 123, "name": "주방세제"},
    {"supercategory": "주방용품", "id": 124, "name": "주방장갑"},
    {"supercategory": "주방용품", "id": 125, "name": "도시락"},
    {"supercategory": "주방용품", "id": 126, "name": "집게"},
    {"supercategory": "주방용품", "id": 127, "name": "채칼"},
    {"supercategory": "주방용품", "id": 128, "name": "컵"},
    {"supercategory": "주방용품", "id": 129, "name": "키친타월"},
    {"supercategory": "주방용품", "id": 130, "name": "텀블러"},
    {"supercategory": "주방용품", "id": 131, "name": "토스터"},
    {"supercategory": "주방용품", "id": 132, "name": "포크"},
    {"supercategory": "주방용품", "id": 133, "name": "포크 스푼"},
    {"supercategory": "주방용품", "id": 134, "name": "후라이팬"},
    {"supercategory": "잡화", "id": 135, "name": "계산기"},
    {"supercategory": "잡화", "id": 136, "name": "고데기"},
    {"supercategory": "잡화", "id": 137, "name": "공"},
    {"supercategory": "잡화", "id": 138, "name": "글러브"},
    {"supercategory": "잡화", "id": 139, "name": "남성 구두"},
    {"supercategory": "잡화", "id": 140, "name": "폼롤러"},
    {"supercategory": "잡화", "id": 141, "name": "다트"},
    {"supercategory": "잡화", "id": 142, "name": "독서대"},
    {"supercategory": "잡화", "id": 143, "name": "돋보기"},
    {"supercategory": "잡화", "id": 144, "name": "라이터"},
    {"supercategory": "잡화", "id": 145, "name": "레고"},
    {"supercategory": "잡화", "id": 146, "name": "루빅 큐브"},
    {"supercategory": "잡화", "id": 147, "name": "리모컨"},
    {"supercategory": "잡화", "id": 148, "name": "마우스"},
    {"supercategory": "잡화", "id": 149, "name": "머리띠"},
    {"supercategory": "잡화", "id": 150, "name": "머리핀"},
    {"supercategory": "잡화", "id": 151, "name": "메이크업 브러쉬"},
    {"supercategory": "잡화", "id": 152, "name": "반지"},
    {"supercategory": "잡화", "id": 153, "name": "마사지볼"},
    {"supercategory": "잡화", "id": 154, "name": "볼링핀"},
    {"supercategory": "잡화", "id": 155, "name": "붕대"},
    {"supercategory": "잡화", "id": 156, "name": "선글라스"},
    {"supercategory": "잡화", "id": 157, "name": "손전등"},
    {"supercategory": "잡화", "id": 158, "name": "슬리퍼"},
    {"supercategory": "잡화", "id": 159, "name": "신발주걱"},
    {"supercategory": "잡화", "id": 160, "name": "안경"},
    {"supercategory": "잡화", "id": 161, "name": "액자"},
    {"supercategory": "잡화", "id": 162, "name": "야구헬멧"},
    {"supercategory": "잡화", "id": 163, "name": "여권"},
    {"supercategory": "잡화", "id": 164, "name": "여성 구두"},
    {"supercategory": "잡화", "id": 165, "name": "열쇠"},
    {"supercategory": "잡화", "id": 166, "name": "온도계"},
    {"supercategory": "잡화", "id": 167, "name": "우산"},
    {"supercategory": "잡화", "id": 168, "name": "운동화"},
    {"supercategory": "잡화", "id": 169, "name": "인형"},
    {"supercategory": "잡화", "id": 170, "name": "자동차장난감"},
    {"supercategory": "잡화", "id": 171, "name": "자석"},
    {"supercategory": "잡화", "id": 172, "name": "장화"},
    {"supercategory": "잡화", "id": 173, "name": "주사위"},
    {"supercategory": "잡화", "id": 174, "name": "주판기"},
    {"supercategory": "잡화", "id": 175, "name": "지갑"},
    {"supercategory": "잡화", "id": 176, "name": "체스판"},
    {"supercategory": "잡화", "id": 177, "name": "체온계"},
    {"supercategory": "잡화", "id": 178, "name": "카메라"},
    {"supercategory": "잡화", "id": 179, "name": "카메라렌즈"},
    {"supercategory": "잡화", "id": 180, "name": "컴퍼스"},
    {"supercategory": "잡화", "id": 181, "name": "크록스"},
    {"supercategory": "잡화", "id": 182, "name": "키링"},
    {"supercategory": "잡화", "id": 183, "name": "타이머"},
    {"supercategory": "잡화", "id": 184, "name": "탬버린"},
    {"supercategory": "잡화", "id": 185, "name": "테이프클리너"},
    {"supercategory": "잡화", "id": 186, "name": "피규어"},
    {"supercategory": "잡화", "id": 187, "name": "필통"},
    {"supercategory": "잡화", "id": 188, "name": "핸드백"},
    {"supercategory": "잡화", "id": 189, "name": "USB"},
    {"supercategory": "욕실용품", "id": 190, "name": "뚜러뻥"},
    {"supercategory": "욕실용품", "id": 191, "name": "비누"},
    {"supercategory": "욕실용품", "id": 192, "name": "비누받침"},
    {"supercategory": "욕실용품", "id": 193, "name": "샤워기"},
    {"supercategory": "욕실용품", "id": 194, "name": "욕실세정제"},
    {"supercategory": "욕실용품", "id": 195, "name": "샴푸"},
    {"supercategory": "욕실용품", "id": 196, "name": "세수대야"},
    {"supercategory": "욕실용품", "id": 197, "name": "스퀴지"},
    {"supercategory": "욕실용품", "id": 198, "name": "치실"},
    {"supercategory": "욕실용품", "id": 199, "name": "치약"},
    {"supercategory": "욕실용품", "id": 200, "name": "칫솔"}
]

def split_annotations_for_dataset(input_base_dir_1, input_base_dir_2, output_vision_file, output_referring_file):
    # Prepare the COCO-style structure for vision annotations
    vision_annotations = {
        "info": {
            "description": "Custom COCO dataset",
            "url": "http://customdataset.org",
            "version": "1.0",
            "year": 2024,
            "contributor": "Custom Dataset Group",
            "date_created": str(datetime.now())
        },
        "images": [],
        "licenses": [
            {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}
        ],
        "annotations": [],
        # "categories": MANUFACT_CATEGORIES
        "categories": INDOOR_CATEGORIES  # 또는 MANUFACT_CATEGORIES
    }

    referring_output = []
    ref_id = 0
    image_id_counter = 0
    annotation_id_counter = 0  # Initialize annotation ID counter

    # 그룹별로 어노테이션 파일 리스트 생성
    ann_list = glob.glob(os.path.join(input_base_dir_1, "group_*", "annotation", "*.json")) + \
               glob.glob(os.path.join(input_base_dir_2, "group_*", "annotation", "*.json"))

    # 그룹별 어노테이션 파일 개수 확인을 위한 딕셔너리
    group_annotation_count = {}

    # 그룹 ID별로 어노테이션 파일을 수집
    group_annotation_files = {}
    for annotation_file in ann_list:
        # 데이터 유형 판별 ('real' 또는 'syn')
        if '/real/' in annotation_file:
            data_type = 'real'
        elif '/synthetic/' in annotation_file:
            data_type = 'syn'
        else:
            continue

        # 그룹 ID 추출
        group_dir = os.path.dirname(os.path.dirname(annotation_file))  # group_000001 directory
        group_num = os.path.basename(group_dir).split('_')[1]  # Extract '000001' from 'group_000001'
        group_id = f"{data_type}_{group_num}"

        # 그룹별로 어노테이션 파일 리스트를 저장
        if group_id not in group_annotation_files:
            group_annotation_files[group_id] = []
        group_annotation_files[group_id].append(annotation_file)

    # 그룹별로 어노테이션 파일 개수 확인 및 그룹 이름 출력
    for group_id, files in group_annotation_files.items():
        if len(files) != 5:
            print(f"그룹 {group_id}의 JSON 파일 개수: {len(files)}")
    print(f"총 그룹 개수: {len(group_annotation_files)}")
    valid_num = 0
    test_num = 0

    # random split
    print("Splitting dataset into train, val, and test sets randomly...")
    idx = 0
    for group_id, files in tqdm(group_annotation_files.items()):
        
        ### To preprocess 1% of the dataset
        # idx += 1
        # if idx%100 != 0:
        #     continue

        split = np.random.choice(['train', 'val', 'test'], p=[0.8, 0.1, 0.1])

        for annotation_file in files:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Construct image file path from annotation file path
            image_file = annotation_file.replace("/labeling_data/", "/source_data/")
            image_file = image_file.replace("/annotation/", "/rgb/")
            image_file = image_file.replace(".json", ".png")

            # Check if image file exists
            if not os.path.exists(image_file):
                print(f"Image file {image_file} does not exist. Skipping.")
                continue

            # Prepare image file name
            file_name = f"{group_id}_{os.path.basename(image_file)}"

            # Prepare vision annotation (COCO format)
            image_entry = {
                "file_name": file_name,
                "id": image_id_counter,  # Use a counter for unique image IDs
                "height": data['images']['height'],
                "width": data['images']['width'],
                "date_captured": str(datetime.now())  # Assuming current date
            }
            vision_annotations['images'].append(image_entry)

            # Process each annotation within the JSON file
            for ann in data['annotations']:
                # Generate a unique annotation ID
                unique_ann_id = annotation_id_counter
                annotation_id_counter += 1  # Increment the counter

                # Vision annotation: Include bbox, segmentation, area, etc.
                try:
                    vision_annotation = {
                        "image_id": image_id_counter,
                        "id": unique_ann_id,
                        "category_id": ann['category_id'],
                        "bbox": ann['bbox'],
                        "segmentation": ann['segmentation'],
                        "area": ann['area'],
                        "iscrowd": ann['iscrowd'],
                    }
                    if ann['bbox'] is None:
                        print(f"Annotation without bbox in file: {annotation_file}")
                        continue
                except KeyError as e:
                    print(f"Error in annotation: Missing key {e} in file {annotation_file}")
                    continue

                # Referring annotation format
                try:
                    sentences = [
                        {"raw": ann['referring_expression'], "sent_id": i, "sent": ann['referring_expression']}
                        for i in range(len([ann['referring_expression']]))
                    ]
                except KeyError:
                    print(f"Missing 'referring_expression' in annotation file: {annotation_file}")
                    continue

                vision_annotations['annotations'].append(vision_annotation)
                valid_num += 1

                referring_annotation = {
                    "ref_id": ref_id,
                    "category_id": ann['category_id'],
                    "image_id": image_id_counter,
                    "file_name": file_name,
                    "ann_id": unique_ann_id,
                    "split": split,
                    "sentences": sentences,
                    "sent_ids": [i for i in range(len(sentences))]
                }
                referring_output.append(referring_annotation)
                ref_id += 1

            image_id_counter += 1
            test_num += 1

            # Move image to the output directory
            output_image_dir = os.path.join(os.path.dirname(output_vision_file), "images")
            os.makedirs(output_image_dir, exist_ok=True)
            shutil.copy(image_file, os.path.join(output_image_dir, file_name))
        
    print("Total valid annotations: ", valid_num)
    print("Total test images: ", test_num)

    # Save the vision annotations to a JSON file
    with open(output_vision_file, 'w', encoding='utf-8') as f:
        json.dump(vision_annotations, f, ensure_ascii=False, indent=4)

    # Save the referring annotations to a pickle file
    with open(output_referring_file, 'wb') as f:
        pickle.dump(referring_output, f)

if __name__ == "__main__":
    # Set the input directories for annotations
    input_dir_1 = "refer/data/labeling_data/real"
    input_dir_2 = "refer/data/labeling_data/synthetic"

    # Set the output file paths
    if not os.path.exists("refer/data/aihub_refcoco_format/indoor"):
        os.makedirs("refer/data/aihub_refcoco_format/indoor")
    output_vision_file = "refer/data/aihub_refcoco_format/indoor/instances.json"
    output_referring_file = "refer/data/aihub_refcoco_format/indoor/refs.p"

    # Call the function to process the dataset
    split_annotations_for_dataset(input_dir_1, input_dir_2, output_vision_file, output_referring_file)


