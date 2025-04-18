# Bash 필수 명령어 정리

## 1. 파일 시스템 탐색
- pwd : 현재 작업 디렉토리 경로 출력
- ls : 디렉토리 내용 출력
  - ls -l : 자세한 정보 포함
  - ls -a : 숨김 파일 포함
- cd <디렉토리> : 디렉토리 이동
  - cd .. : 상위 디렉토리로 이동
  - cd ~ : 홈 디렉토리로 이동

## 2. 파일 및 디렉토리 관리
- mkdir <디렉토리명> : 디렉토리 생성
- touch <파일명> : 빈 파일 생성
- cp <원본> <대상> : 파일/디렉토리 복사
- mv <원본> <대상> : 파일/디렉토리 이동 또는 이름 변경
- rm <파일명> : 파일 삭제
  - rm -r <디렉토리> : 디렉토리와 내용 삭제
  - rm -f : 강제 삭제

## 3. 파일 내용 확인
- cat <파일명> : 파일 내용 출력
- less <파일명> : 파일 내용 페이지 단위로 확인
- head <파일명> : 파일 앞부분 출력
- tail <파일명> : 파일 뒷부분 출력
- grep <패턴> <파일명> : 파일에서 패턴 검색

## 4. 권한 관리
- chmod : 파일 권한 변경
  - chmod 755 <파일명> : 실행 권한 부여
  - chmod +x <파일명> : 실행 권한 추가
- chown : 파일 소유자 변경

## 5. 프로세스 관리
- ps : 실행 중인 프로세스 목록
- kill <PID> : 프로세스 종료
- top : 시스템 상태 모니터링

## 6. 기타 유용한 명령어
- echo : 텍스트 출력
- clear : 터미널 화면 지우기
- history : 명령어 실행 기록 확인
- | (파이프) : 명령어 출력을 다른 명령어의 입력으로 전달
- > : 출력 리다이렉션 (파일로 저장)
- >> : 출력 추가 리다이렉션 (파일에 추가)
