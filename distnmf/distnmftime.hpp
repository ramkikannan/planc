/* Copyright 2016 Ramakrishnan Kannan */

#ifndef DISTNMF_DISTNMFTIME_HPP_
#define DISTNMF_DISTNMFTIME_HPP_

namespace planc {

class DistNMFTime {
 private:
  double m_duration;
  double m_compute_duration;
  double m_communication_duration;
  double m_allgather_duration;
  double m_allreduce_duration;
  double m_reducescatter_duration;
  double m_gram_duration;
  double m_mm_duration;
  double m_nnls_duration;
  double m_err_compute_duration;
  double m_err_communication_duration;

 public:
  DistNMFTime(double d, double compute_d, double communication_d,
              double err_comp, double err_comm)
      : m_duration(d),
        m_compute_duration(compute_d),
        m_communication_duration(communication_d),
        m_err_compute_duration(err_comp),
        m_err_communication_duration(err_comm) {}
  DistNMFTime(double d, double compute_d, double communication_d,
              double allgather_d, double allreduce_d, double reducescatter_d,
              double gram_d, double mm_d, double nnls_d, double err_comp,
              double err_comm)
      : m_duration(d),
        m_compute_duration(compute_d),
        m_communication_duration(communication_d),
        m_allgather_duration(allgather_d),
        m_allreduce_duration(allreduce_d),
        m_reducescatter_duration(reducescatter_d),
        m_gram_duration(gram_d),
        m_mm_duration(mm_d),
        m_nnls_duration(nnls_d),
        m_err_compute_duration(err_comp),
        m_err_communication_duration(err_comm) {}
  DistNMFTime(double d, double compute_d, double communication_d, double gram_d,
              double mm_d, double nnls_d, double err_comp, double err_comm)
      : m_duration(d),
        m_compute_duration(compute_d),
        m_communication_duration(communication_d),
        m_gram_duration(gram_d),
        m_mm_duration(mm_d),
        m_nnls_duration(nnls_d),
        m_err_compute_duration(err_comp),
        m_err_communication_duration(err_comm) {}

  const double duration() const { return m_duration; }
  const double compute_duration() const { return m_compute_duration; }
  const double communication_duration() const {
    return m_communication_duration;
  }
  const double allgather_duration() const { return m_allgather_duration; }
  const double allreduce_duration() const { return m_allreduce_duration; }
  const double reducescatter_duration() const {
    return m_reducescatter_duration;
  }
  const double gram_duration() const { return m_gram_duration; }
  const double mm_duration() const { return m_mm_duration; }
  const double nnls_duration() const { return m_nnls_duration; }
  const double err_compute_duration() const { return m_err_compute_duration; }
  const double err_communication_duration() const {
    return m_err_communication_duration;
  }
  void duration(double d) { m_duration += d; }
  void compute_duration(double d) { m_compute_duration += d; }
  void communication_duration(double d) { m_communication_duration += d; }
  void allgather_duration(double d) { m_allgather_duration += d; }
  void allreduce_duration(double d) { m_allreduce_duration += d; }
  void reducescatter_duration(double d) { m_reducescatter_duration += d; }
  void gram_duration(double d) { m_gram_duration += d; }
  void mm_duration(double d) { m_mm_duration += d; }
  void nnls_duration(double d) { m_nnls_duration += d; }
  void err_compute_duration(double d) { m_err_compute_duration += d; }
  void err_communication_duration(double d) {
    m_err_communication_duration += d;
  }
};

}  // namespace planc

#endif  // DISTNMF_DISTNMFTIME_HPP_
