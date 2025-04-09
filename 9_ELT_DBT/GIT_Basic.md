# Git 필수 명령어 정리

## 1. 기본 설정
- git config --global user.name "사용자이름"
- git config --global user.email "이메일주소"

## 2. 저장소 생성 및 복제
- git init : 새로운 Git 저장소 생성
- git clone <url> : 원격 저장소 복제

## 3. 변경사항 관리
- git status : 현재 상태 확인
- git add <파일명> : 특정 파일 스테이징
- git add . : 모든 변경사항 스테이징
- git commit -m "커밋메시지" : 변경사항 커밋
- git commit --amend : 마지막 커밋 수정

## 4. 브랜치 관리
- git branch : 브랜치 목록 확인
- git branch <브랜치명> : 새 브랜치 생성
- git checkout <브랜치명> : 브랜치 전환
- git merge <브랜치명> : 현재 브랜치에 다른 브랜치 병합
- git branch -d <브랜치명> : 브랜치 삭제

## 5. 원격 저장소 관리
- git remote add origin <url> : 원격 저장소 추가
- git push origin <브랜치명> : 원격 저장소에 푸시
- git pull origin <브랜치명> : 원격 저장소에서 풀
- git fetch : 원격 저장소 정보 가져오기

## 6. 변경사항 확인
- git log : 커밋 히스토리 조회
- git diff : 변경사항 비교
- git show <커밋해시> : 특정 커밋 상세 정보

## 7. 되돌리기
- git reset --hard <커밋해시> : 특정 커밋으로 완전히 되돌리기
- git revert <커밋해시> : 특정 커밋 취소하는 새로운 커밋 생성
- git checkout -- <파일명> : 파일 변경사항 취소
