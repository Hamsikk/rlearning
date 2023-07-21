==================repository==================

git@github.com:HanByel/FEV_Cluster.git

*******수정하기 전 git의 프로그램을 download 먼저 하고 수정 후 업로드

전체 내용 다운로드

수정

업로드

****다른 사람이 수정하면 충돌로 인하여 업로드가 안됨

==================setting(윈도우)==================

하기 사이트에서 프로그램 설치

https://git-scm.com/

cd 원하는 디렉토리

mkdir 폴더이름

cd 폴더이름

git init

git remote add origin git@github.com:HanByel/FEV_Cluster.git

git remote -v

==================setting(리눅스)==================

mkdir ~/MyProject

cd ~/myproject

git init

git remote add origin git@github.com:HanByel/FEV_Cluster.git

git remote -v

==================download==================

git pull origin master

git status

==================upload==================

git add .

*********커밋중요**********

git commit -m "
- [EVSC][priv] practice
- name : JUNSIK HAM
- date : 2023-07-19
"

git push --all

==================편하게 셋팅하는 방법==================

git config --global user.name "git이름"

git config --global user.email "깃허브 메일주소" // 매번 물어보는 귀찮음을 피하기 위해 설정.

Git 에서 비밀번호 물어보지 않게 하기

Git 를 사용할 때 git pull 등의 명령어를 실행할 때 비밀번호를 물어보게 될 때 매번 물어보지 않게 하는 방법을 안내한다.

두가지 방법이 있는데

repository url 에서 비밀번호는 입력하는 방법

캐시를 설정하는 방법이 있다.

git clone 을 할 때 다음과 같이 비밀번호를 미리 입력한다면 비밀번호를 물어보지 않는다. 이렇게 이용할 수도 있으나 추천 하지는 않는다.

git clone https://myid:mypassword@bitbucket.org/myid/myapp.git

캐시를 설정하여 특정시간동안 비밀번호를 다시 물어보지 않게끔 할 수 있다.

여기서 –global 옵션과 함께 명령어를 실행한다면 모든 계정에 대해서 캐싱을 한다는 의미이다. 만약 특정 git 디렉토리만 설정한다면 –global 옵션을 제거해 주면 된다.

git config --global credential.helper cache

디폴트는 900초(15분) 동안 캐시를 저장하므로 15분이 지나고 다시 시도하면 비밀번호를 다시 물어보게 된다.

넉넉하게 타임아웃을 설정할 수 있는데 만약 10일간 캐시를 설정한다고 한다면 다음과 같이 입력하면 된다.

git config --global credential.helper 'cache --timeout=864000'

이제 더 이상 비밀번호를 물어보지 않는다. 야호!
