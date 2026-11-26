# Backend image: builds the Spring Boot jar with Maven, then runs it on a
# slim JRE. The trained models (src/main/resources/models/*.ser) and
# metrics.json are baked into the jar at build time, so the container does
# NOT retrain on startup. The dataset CSV is mounted read-only at
# /app/data for POST /api/retrain (see docker-compose.yml).
FROM maven:3.9-eclipse-temurin-25 AS build
WORKDIR /build
COPY pom.xml .
COPY src ./src
# Offline-friendly: dependencies are resolved from Maven Central at image
# build time (the machine building the image needs internet).
RUN mvn -q package -DskipTests

FROM eclipse-temurin:25-jre
WORKDIR /app
COPY --from=build /build/target/pcod-prediction-backend-1.0.0.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
