FROM ubuntu:24.04

# Устанавливаем переменные среды
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Обновление и установка по шагам (с сохранением кэша)
RUN apt update && apt upgrade -y && apt install -y build-essential
RUN apt install -y libssl-dev
RUN apt install -y libboost-all-dev
RUN apt install -y libeigen3-dev
RUN apt install -y cmake
RUN apt install -y bash
RUN apt install -y libopencv-dev
RUN apt clean

# Рабочая директория root-пользователя
WORKDIR /root

# Запуск контейнера в интерактивной оболочке
CMD ["/bin/bash"]

