#pragma once
#include <ctime>
#include <cmath>
#include <cstdint>

typedef struct /* time struct */
{
    time_t time;                /* time (s) expressed by standard time_t */
    double sec;                 /* fraction of second under 1 s */
    long long ToUnix_us() const // 精确到0.1ms
    {
        return time * 1000 * 1000 + static_cast<long long>(round(sec * 1000 * 10) * 100);
    }
} gtime_t;

gtime_t timeadd(gtime_t t, double sec);

double timediff(gtime_t t1, gtime_t t2);

gtime_t epoch2time(const double *ep);

gtime_t utc2gpst(gtime_t t);

double time2gpst(gtime_t t, int *week);

// t: 毫秒
void time2WS(uint64_t t, uint16_t &week, uint32_t &ms);

// GPS周秒转换成gtime_t结构
gtime_t gpst2time(int week, double sec);

gtime_t gpst2utc(gtime_t t);

void weeksconvert(uint64_t time, uint16_t &week, uint32_t &tow_ms);

// 将unix的longlong时间转换成GPS周秒
double Unix2GPST(uint64_t time, uint16_t *week = NULL, bool b10ms = false);

// GPS周秒转long long时间，us
long long GPSWS2Unix(const uint16_t week, const double sec);

// 根据接收到时(ms)计算周，一周的第几天，再根据当天秒(时分秒)计算unix时间
long long SecOfDay2Unix(const long long rt, const double secofday);

double HHHMMSS2Sec(const double t);